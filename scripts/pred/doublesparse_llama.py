import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import gc

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from math import sqrt

def pseudo_quantize(tensor, q_bit):
    max_quant = 2 ** q_bit - 1

    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    scale = max_quant / range_val
    quantized = torch.round((tensor - min_val) * scale).clamp(0, max_quant)

    dequantized = quantized / scale + min_val

    return dequantized

class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # channel config
        self.sorted_channel = None

        # heavy const
        self.heavy_const = 2048
        self.group_factor = 1
        self.label_bits = 16
        self.init_const = 128
        self.local_const = 128

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # stats
        self.collect_stats = False
        self.overlaps = {}
        self.recalls = {}
        self.precision = {}
        self.print_offloading_flag = False
        self.offloading_length = 250000


    def __repr__(self):
        return f"{super().__repr__()}\nSparsification Setting(topk:{self.heavy_const},channel_reduction:{self.group_factor},label_bits:{self.label_bits},edge:{self.init_const, self.local_const},stats:{self.collect_stats})"
        
    def compute_stats(self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ):
        ''' independent computation from forward pass to enable logging even when we do full attention
             Expects that the KV Cache is already on the GPU and that handling is outside the function
        '''

        if (past_key_value is None or 
            len(past_key_value.key_cache) <= self.layer_idx or
            past_key_value.key_cache[self.layer_idx].shape[-2]  % 1024 != 0):
            return 

        # prepare keys and queries.
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
           
            
        if position_embeddings is None:
            print(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(query_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = past_key_value.key_cache[self.layer_idx] # keys already appended in cache
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        kv_seq_len = key_states.shape[2]


        # target
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 
        # causal_mask + recent budget + init budget
        causal_heavy_recent_mask = torch.tril(torch.ones(q_len,kv_seq_len,device=attn_weights.device), diagonal=kv_seq_len-q_len-self.local_const).bool()
        causal_heavy_recent_mask[:,:self.init_const] = False
        
        attn_weights.masked_fill_(torch.logical_not(causal_heavy_recent_mask), torch.finfo(attn_weights.dtype).min)
        target = torch.zeros_like(attn_weights)
        _,idx = torch.topk(attn_weights, dim=-1, k=32) # [B,A,S,T]
        view_idx = idx.view(-1,idx.shape[-1])
        view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * attn_weights.shape[-1]
        target.view(-1)[view_idx.view(-1)] = 1.0


        # predicted # dont want it to be conditioned on retrieval algorithm .. since I just want to measure the quality
        assert self.head_dim % self.group_factor == 0
        assert self.sorted_channel is not None
        sorted_query_states = query_states.transpose(1,2)
        sorted_key_states = key_states.transpose(1,2)
        sorted_query_states = torch.gather(sorted_query_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, q_len, -1, -1)).transpose(1,2)
        sorted_key_states = torch.gather(sorted_key_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, kv_seq_len, -1, -1)).transpose(1,2)
        # outlier channel only
        outlier_num = self.head_dim // self.group_factor
        grouped_query = sorted_query_states[:,:,:,:outlier_num]
        grouped_key = sorted_key_states[:,:,:,:outlier_num]
        # quantization
        if self.label_bits < 16:
            grouped_query = pseudo_quantize(grouped_query, self.label_bits)
            grouped_key = pseudo_quantize(grouped_key, self.label_bits)
        grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // self.group_factor)
        span = grouped_attn_weights        
        span.masked_fill_(torch.logical_not(causal_heavy_recent_mask),torch.finfo(span.dtype).min)
        pred = torch.zeros_like(span)
        _,idx = torch.topk(span, dim=-1, k=kv_seq_len // 16) # [B,A,S,T]
        view_idx = idx.view(-1,idx.shape[-1])
        view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * span.shape[-1]
        pred.view(-1)[view_idx.view(-1)] = 1.0

        # stats.
        overlap = pred * target
        overlap_ratio = torch.sum(overlap, dim=-1) / torch.sum(target, dim=-1)
        
        ## add to collection
        if kv_seq_len not in self.overlaps.keys():
            self.overlaps[kv_seq_len] = [0,0,0,0,0] # sum, sqsum, ct, mean, std
            
        self.overlaps[kv_seq_len][0] += overlap_ratio.sum().item()
        self.overlaps[kv_seq_len][1] += torch.square(overlap_ratio).sum().item()
        self.overlaps[kv_seq_len][2] += overlap_ratio.numel()
        self.overlaps[kv_seq_len][3] = self.overlaps[kv_seq_len][0] / self.overlaps[kv_seq_len][2]
        self.overlaps[kv_seq_len][4] = sqrt(self.overlaps[kv_seq_len][1] / self.overlaps[kv_seq_len][2] - self.overlaps[kv_seq_len][3]**2)

        if self.layer_idx == 17:
            print(self.overlaps)        

        
    def ensure_gpu(self, past_key_value, device):
        if (past_key_value is not None 
            and  len(past_key_value.key_cache) > self.layer_idx 
            and (not past_key_value.key_cache[self.layer_idx].is_cuda)):
            #print("onboarding layer", self.layer_idx)
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].to(device)
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].to(device)

    def offload_if_necessary_cpu(self, past_key_value):
        if (past_key_value is not None 
            and  len(past_key_value.key_cache) > self.layer_idx  
            and past_key_value.key_cache[self.layer_idx].shape[2] >=self.offloading_length):
            if self.print_offloading_flag == False:
                print("OFFLOADING ENABLED >>")
                self.print_offloading_flag = True
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].cpu()
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].cpu()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # if self.config.num_hidden_layers != 32:
        #     gc.collect()
        #     torch.cuda.empty_cache()
        self.ensure_gpu(past_key_value, hidden_states.device)
        assert (not(q_len < 130 and q_len > 125)) # making sure that the correct offset is used
        if q_len > 1:
            return_value =  self.flash_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if self.collect_stats:
                self.compute_stats(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    position_embeddings
                    )
            self.offload_if_necessary_cpu(past_key_value)
            return return_value

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
        if position_embeddings is None:
            print(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)   
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)


        # group_factor = 8
        assert self.head_dim % self.group_factor == 0

        kv_seq_len = key_states.shape[-2]
        if self.sorted_channel is not None:
            sorted_query_states = query_states.transpose(1,2)
            sorted_key_states = key_states.transpose(1,2)
            sorted_query_states = torch.gather(sorted_query_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, q_len, -1, -1)).transpose(1,2)
            sorted_key_states = torch.gather(sorted_key_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, kv_seq_len, -1, -1)).transpose(1,2)

            # grouped by mean
            # grouped_query = sorted_query_states.reshape(bsz, self.num_heads, q_len, self.head_dim // group_factor, group_factor).sum(dim=-1) / group_factor
            # grouped_key = sorted_key_states.reshape(bsz, self.num_heads, kv_seq_len, self.head_dim // group_factor, group_factor).sum(dim=-1) / group_factor
            # grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // group_factor)

            # outlier channel only
            outlier_num = self.head_dim // self.group_factor
            grouped_query = sorted_query_states[:,:,:,:outlier_num]
            grouped_key = sorted_key_states[:,:,:,:outlier_num]


            # quantization
            if self.label_bits < 16:
                grouped_query = pseudo_quantize(grouped_query, self.label_bits)
                grouped_key = pseudo_quantize(grouped_key, self.label_bits)


            grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // self.group_factor)

            # precision problem??
        else:
            grouped_query = query_states.reshape(bsz, self.num_heads, q_len, self.head_dim // self.group_factor, self.group_factor).sum(dim=-1) / self.group_factor
            grouped_key = key_states.reshape(bsz, self.num_heads, kv_seq_len, self.head_dim // self.group_factor, self.group_factor).sum(dim=-1) / self.group_factor
            grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // self.group_factor)

        # assert torch.allclose(attn_weights, grouped_attn_weights, atol=0.001), f"{torch.nonzero(torch.abs(attn_weights - grouped_attn_weights))}"


        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # NOTE: transformers 4.44 doesn't provide attention_mask for LlamaAttention??
        # print(f"attention mask is {attention_mask}.")
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            grouped_attn_weights = grouped_attn_weights + attention_mask
        elif q_len == kv_seq_len:
            boolean_mask = torch.tril(torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=attn_weights.device))
            attention_mask = torch.zeros(q_len, kv_seq_len, dtype=torch.float16, device=attn_weights.device)
            attention_mask = attention_mask.masked_fill(boolean_mask == False, float('-inf')).view(1, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask
            grouped_attn_weights = grouped_attn_weights + attention_mask

        h2_mask = torch.zeros_like(attn_weights).bool()
        # heavy_const = 256
        # [bs, num_heads, q_len, kv_len] -> [bs, num_heads, q_len, heavy_const]
        # sorted_weights, indices = attn_weights.sort(dim=-1, descending=True)

        # recent and init mask on weights to remove from topk computation
        assert q_len == 1 # setting recent etc only works for q_len = 1 or else if you want batched processing then use 

        if self.heavy_const > 1.0:
            heavy_const = int(self.heavy_const)
        else:
            heavy_const = int(kv_seq_len * self.heavy_const)

        grouped_attn_weights[:,:,:,:self.init_const]  = float('-inf')
        grouped_attn_weights[:,:,:,-self.local_const:]  = float('-inf')
        values, indices = grouped_attn_weights.sort(dim=-1, descending=True)
        discard_indices = indices[:, :, :, heavy_const:]
        h2_mask.scatter_(3, discard_indices, 1)

        # recent and local
        h2_mask[:,:,:,:self.init_const] = False
        h2_mask[:,:,:,-self.local_const:] = False
        attn_weights.masked_fill_(h2_mask, float('-inf'))
        
    
        # # free gpu memory
        # if self.config.num_hidden_layers != 32:
        #     h2_mask = None
        #     grouped_attn_weights = None
        #     indices = None
        #     discard_indices = None
        #     grouped_query = None
        #     grouped_key = None
        #     sorted_query_states = None
        #     sorted_key_states = None
        #     query_states = None
        #     key_states = None
        #     gc.collect()
        #     torch.cuda.empty_cache()

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



def convert_kvcache_heavy_recent(model, config, heavy_const, group_factor, label_bits, init_const, local_const, collect_stats):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_heavy_recent(module, config, heavy_const, group_factor, label_bits, init_const, local_const, collect_stats)

        if isinstance(module, LlamaAttention):
            device = next(module.parameters()).device
            new_module = LlamaAttention_heavy_hitter(config, module.layer_idx).bfloat16().to(device)
            new_module.load_state_dict(module.state_dict())
            new_module.heavy_const = heavy_const
            new_module.init_const = init_const
            new_module.local_const = local_const
            new_module.group_factor = group_factor
            new_module.label_bits = label_bits
            new_module.collect_stats = collect_stats
            model._modules[name] = new_module
            model._modules[name].flash_forward = module.forward

    return model


def convert_channel_config(model, channel_config, selected_channel="k"):

    selected_channel = "." + selected_channel + "_proj"

    for name, module in model.named_modules():

        if isinstance(module, LlamaAttention_heavy_hitter):
            device = next(module.parameters()).device
            module.sorted_channel = torch.tensor(channel_config[name + selected_channel]).to(device)

    return model


def change_heavy_const(model, heavy_const=128, group_factor=4, label_bits=4):

    for name, module in model.named_modules():

        if isinstance(module, LlamaAttention_heavy_hitter):
            
            module.heavy_const = heavy_const
            module.group_factor = group_factor
            module.label_bits = label_bits

    return model

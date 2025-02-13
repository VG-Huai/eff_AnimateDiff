from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
import xformers

from einops import rearrange, repeat
import math


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels,
    motion_module_type: str, 
    motion_module_kwargs: dict
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs,)    
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )
        
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None, 
                mask_dict=None, attn_bias_dict=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask, mask_dict=mask_dict, attn_bias_dict=attn_bias_dict)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)    
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, mask_dict=None, attn_bias_dict=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, 
                                  mask_dict=mask_dict, attn_bias_dict=attn_bias_dict)
        
        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, mask_dict=None, attn_bias_dict=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                video_length=video_length,
                mask_dict=mask_dict,
                attn_bias_dict=attn_bias_dict
            ) + hidden_states
        if mask_dict is not None:
            # indices = mask['indices']
            indices1 = mask_dict['temporal']['indices1']
            indices2 = indices1.squeeze()
            actual_indices = mask_dict['temporal']['actual_indices']
            # mask = mask_dict['mask']
            norm_hidden_states = self.ff_norm(hidden_states)
            cat_x = self.get_filtered_tensor(norm_hidden_states, indices1)
            x = self.token_reuse(hidden_states, self.ff(cat_x), indices2, actual_indices) + hidden_states
            hidden_states = x
        else:
            hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output
    
    def token_reuse(self, x_in, x_out, indices, actual_indices):
        # temporal version
        out = torch.zeros_like(x_in)
        out = out.reshape(-1, out.shape[-1])
        out.index_put_((indices,), x_out.squeeze())
        out = out.reshape(x_in.shape[0], x_in.shape[1], -1) 
        actual_indices = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        
        frames_num = actual_indices.shape[1]
        _, M, K = out.shape
        out = out.reshape(-1, frames_num, M, K)
        out = out.permute(0, 2, 1, 3) # B, T, M, K -> B, M, T, K
        out = out.reshape(-1, frames_num, K) # B, M, T, K -> B * M, T, K
        out = out.gather(1, actual_indices) # B * M, T, K -> B * M, T, K
        out = out.reshape(-1, M, frames_num, K) # B * M, T, K -> B, M, T, K
        out = out.permute(0, 2, 1, 3) # B, M, T, K -> B, T, M, K
        out = out.reshape(-1, M, K)
        # if mode == 'spatial':
        #     out = out.permute(1, 0, 2)
        #     out = out.gather(1, actual_indices).permute(1, 0, 2)
        # else:
        #     out = out.gather(1, actual_indices)
        return out
    def get_filtered_tensor(self, x, indices):
        x = x.reshape(-1, x.shape[-1])
        x = torch.index_select(x, 0, indices.squeeze())
        x = x.reshape(1, -1, x.shape[-1])
        return x


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 32,            
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") else None

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, mask_dict=None, attn_bias_dict=None, 
                mode='temporal', use_bias=False):
        if mask_dict is None:
            # print("mask_dict is None")
            batch_size, sequence_length, _ = hidden_states.shape

            if self.attention_mode == "Temporal":
                d = hidden_states.shape[1]
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                
                if self.pos_encoder is not None:
                    hidden_states = self.pos_encoder(hidden_states)
                
                encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
            else:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            dim = query.shape[-1]
            query = self.reshape_heads_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, attention_mask)
                else:
                    hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if self.attention_mode == "Temporal":
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        else:
            heads_num = self.heads
            
            
            if self.attention_mode == "Temporal":
                d = hidden_states.shape[1]
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                
                if self.pos_encoder is not None:
                    hidden_states = self.pos_encoder(hidden_states)
                
                encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
            else:
                raise NotImplementedError
            B, M, C = hidden_states.shape
            
            # eff mode
            if mask_dict is not None:
                # indices = mask['indices']
                indices1 = mask_dict[mode]['indices1']
                indices2 = indices1.squeeze()
                actual_indices = mask_dict[mode]['actual_indices']
                mask = mask_dict['mask']
            if mask is not None:
                # time_stamp = time.time()
                mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
                mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)') 
            if attn_bias_dict is not None:
                if encoder_hidden_states is not None:
                    attention_mask = attn_bias_dict[mode]['cross']
                else:
                    attention_mask = attn_bias_dict[mode]['self']

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
        
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            # query = query.view(B, M, heads_num, C // heads_num)
            # key = key.view(B, key_tokens, heads_num, C // heads_num)
            # value = value.view(B, key_tokens, heads_num, C // heads_num)
            
            cat_q = query.reshape(-1, C)
            cat_q = torch.index_select(cat_q, 0, indices1.squeeze()) 
            cat_q = cat_q.reshape(1, -1, heads_num, C // heads_num)
            
            cat_k = key.reshape(1, -1, heads_num, C // heads_num)
            cat_v = value.reshape(1, -1, heads_num, C // heads_num)
            
            out_with_bias = xformers.ops.memory_efficient_attention(
                cat_q, cat_k, cat_v, attn_bias=attention_mask, scale=self.scale
            )
            out_with_bias = out_with_bias.reshape(1, out_with_bias.shape[1], C) # B M H K ---> B M HK
            out_with_bias = out_with_bias.to(query.dtype)
            
            if use_bias:
                out_with_bias = self.to_out[0](out_with_bias)
                out_with_bias = self.to_out[1](out_with_bias)
                return out_with_bias
            
            out = torch.zeros_like(hidden_states)
            out = out.reshape(-1, C)
            out.index_put_((indices2,), out_with_bias.squeeze())
            
            # token reuse, to be 
            if mask is not None:
                actual_indices1 = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
                if mode == 'spatial':
                    # x: (B T) S C
                    out = out.reshape(mask.shape[0], mask.shape[1], -1)  # b * t, h * w, c
                    out = out.permute(1, 0, 2)
                    out = out.gather(1, actual_indices1).permute(1, 0, 2)

                else:
                    # x: (B S) T C
                    # out = out.reshape(-1, video_length, C)
                    out = out.reshape(B, M, C)  # b * h * w, t, c
                    out = out.gather(1, actual_indices1)

            hidden_states = out
            hidden_states = hidden_states.reshape(B, M, C)
            
            # assert (hidden_states != self.token_reuse(hidden_states, out_with_bias, indices2, actual_indices, mode)).sum() == 0

            # linear proj, for stride mismatch
            if mode == 'spatial':
                hidden_states = self.to_out[0](hidden_states.permute(1, 0, 2))
                # dropout
                hidden_states = self.to_out[1](hidden_states).permute(1, 0, 2)
            else:
                hidden_states = self.to_out[0](hidden_states)
                hidden_states = self.to_out[1](hidden_states)
            if self.attention_mode == "Temporal":
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        return hidden_states
    
    def token_reuse(self, x_in, x_out, indices, actual_indices, mode):
        out = torch.zeros_like(x_in)
        out = out.reshape(-1, out.shape[-1])
        out.index_put_((indices,), x_out.squeeze())
        out = out.reshape(x_in.shape[0], x_in.shape[1], -1) 
        actual_indices = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        if mode == 'spatial':
            out = out.permute(1, 0, 2)
            out = out.gather(1, actual_indices).permute(1, 0, 2)
        else:
            out = out.gather(1, actual_indices)
        return out
    def get_filtered_tensor(self, x, indices):
        x = x.reshape(-1, x.shape[-1])
        x = torch.index_select(x, 0, indices.squeeze())
        x = x.reshape(1, -1, x.shape[-1])
        return x
    def max_diff(self, A, B, name_A, name_B):
        diff = (A - B).abs().max().item()
        print(f"Max error between {name_A} and {name_B}: {diff:.6f}")
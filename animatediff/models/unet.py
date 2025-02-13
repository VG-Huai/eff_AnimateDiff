# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import json
import pdb

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from .resnet import InflatedConv3d, InflatedGroupNorm
from einops import rearrange
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
import torch.nn.functional as F

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,      
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        
        use_inflated_groupnorm=False,
        
        # Additional
        use_motion_module              = False,
        motion_module_resolutions      = ( 1,2,4,8 ),
        motion_module_mid_block        = False,
        motion_module_decoder_only     = False,
        motion_module_type             = None,
        motion_module_kwargs           = {},
        unet_use_cross_frame_attention = False,
        unet_use_temporal_attention    = False,
    ):
        super().__init__()
        
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2 ** i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                
                use_motion_module=use_motion_module and (res in motion_module_resolutions) and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        
        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,

                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,

        # support controlnet
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,

        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers
        input_batch_size, c, frame, h, w = sample.shape
        
        # filter toekn percentage
        if timestep <= 500:
            keep_idxs = self.batched_find_idxs_to_keep(sample, threshold=0.4, tubelet_size=1, patch_size=1)
        else:
            keep_idxs = self.batched_find_idxs_to_keep(sample, threshold=0.3, tubelet_size=1, patch_size=1)
        # keep_idxs = self.batched_find_idxs_to_keep(sample, threshold=0.5, tubelet_size=1, patch_size=1)
        print('------------------')
        total_tokens = keep_idxs.numel()
        filtered_tokens = (keep_idxs == 0).sum().item()
        filtered_percentage = 100.0 * filtered_tokens / total_tokens
        print('timestep:', timestep)
        print(f"Mask Filtering: {filtered_percentage:.2f}% tokens filtered")    
        
        # get all the mask and attn_bias
        mask_dict = {}
        attn_bias_dict = {}
        
        cur_h, cur_w = sample.shape[-2:]
        org_h, org_w = cur_h, cur_w
        mask = keep_idxs
        for i in range(4):
            mask_dict[(cur_h, cur_w)] = {}
            mask_dict[(cur_h, cur_w)]['mask'] = mask
            mask_dict[(cur_h, cur_w)]['spatial'] = {}
            mask_dict[(cur_h, cur_w)]['temporal'] = {}
            mask_dict = self.compute_mask_dict_spatial(mask_dict, cur_h, cur_w)
            mask_dict = self.compute_mask_dict_temporal(mask_dict, cur_h, cur_w)
            # spatial mode
            seq_len_q_spatial = cur_h * cur_w # h * w
            seq_len_q_temporal = sample.shape[2] # t
            if encoder_hidden_states is not None:
                seq_len_kv_spatial = encoder_hidden_states.shape[1]
                seq_len_kv_temporal = encoder_hidden_states.shape[1]
            else:
                seq_len_kv_spatial = seq_len_q_spatial
                seq_len_kv_temporal = seq_len_q_temporal
            attn_bias_dict[(cur_h, cur_w)] = {}
            attn_bias_dict[(cur_h, cur_w)]['spatial'] = {}
            attn_bias_dict[(cur_h, cur_w)]['temporal'] = {}
            attn_bias_dict[(cur_h, cur_w)]['spatial']['self'], _, _ = self.create_block_diagonal_attention_mask(mask, seq_len_q_spatial, mode='spatial')
            attn_bias_dict[(cur_h, cur_w)]['temporal']['self'], _, _ = self.create_block_diagonal_attention_mask(mask, seq_len_q_temporal, mode='temporal')
            attn_bias_dict[(cur_h, cur_w)]['spatial']['cross'], _, _ = self.create_block_diagonal_attention_mask(mask, seq_len_kv_spatial, mode='spatial')
            attn_bias_dict[(cur_h, cur_w)]['temporal']['cross'], _, _ = self.create_block_diagonal_attention_mask(mask, seq_len_kv_temporal, mode ='temporal')
            # attn_bias_dict[(cur_h, cur_w)]['spatial']['self'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_q)
            # attn_bias_dict[(cur_h, cur_w)]['temporal']['self'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_q)
            # attn_bias_dict[(cur_h, cur_w)]['spatial']['cross'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_kv)
            # attn_bias_dict[(cur_h, cur_w)]['temporal']['cross'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_kv)
            cur_h, cur_w = cur_h//2, cur_w//2
            mask = self.resize_mask(mask, cur_h, cur_w)        
        
        # mask_dict = None
        # attn_bias_dict = None
        
        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    mask_dict = mask_dict,
                    attn_bias_dict = attn_bias_dict,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, 
                                                       mask_dict = mask_dict, attn_bias_dict = attn_bias_dict)

            down_block_res_samples += res_samples

        # support controlnet
        down_block_res_samples = list(down_block_res_samples)
        if down_block_additional_residuals is not None:
            for i, down_block_additional_residual in enumerate(down_block_additional_residuals):
                if down_block_additional_residual.dim() == 4: # boardcast
                    down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
                down_block_res_samples[i] = down_block_res_samples[i] + down_block_additional_residual

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
            mask_dict = mask_dict, attn_bias_dict = attn_bias_dict,
        )

        # support controlnet
        if mid_block_additional_residual is not None:
            if mid_block_additional_residual.dim() == 4: # boardcast
                mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
            sample = sample + mid_block_additional_residual

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    mask_dict = mask_dict,
                    attn_bias_dict = attn_bias_dict,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size, encoder_hidden_states=encoder_hidden_states,
                    mask_dict = mask_dict, attn_bias_dict = attn_bias_dict,
                )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
    def create_block_diagonal_attention_mask(self, mask, kv_seqlen, mode='spatial'):
        """
        将 mask 和 kv_seqlen 转换为 BlockDiagonalMask, 用于高效的注意力计算。
        
        Args:
            mask (torch.Tensor): 输入的掩码，标记哪些 token 应该被忽略。
            kv_seqlen (torch.Tensor): 键/值的序列长度。
            heads (int): 注意力头的数量。

        Returns:
            BlockDiagonalPaddedKeysMask: 转换后的注意力掩码，用于高效的计算。
        """
        # 计算 q_seqlen: 通过 mask 来提取有效的查询 token 数量
        mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        if mode == 'spatial':
            mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)')
        else:
            mask = rearrange(mask, 'b 1 t h w -> (b h w) (t)')
        
        q_seqlen = mask.sum(dim=-1)  # 计算每个批次中有效的查询 token 数量
        q_seqlen = q_seqlen.tolist()
        
        kv_seqlen = [kv_seqlen] * len(q_seqlen)  # 重复 kv_seqlen 次

        # 生成 BlockDiagonalPaddedKeysMask
        attn_bias = BlockDiagonalMask.from_seqlens(
            q_seqlen,  
            kv_seqlen=kv_seqlen,  # 键/值的序列长度
        )
        
        return attn_bias, q_seqlen, kv_seqlen    
    
    def compute_mask_dict_spatial(self, mask_dict, cur_h, cur_w):
        mask = mask_dict[(cur_h, cur_w)]['mask']
        indices = []
        _mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        indices1 = torch.nonzero(_mask.reshape(1, -1).squeeze(0))
        _mask = rearrange(_mask, 'b 1 t h w -> (b t) (h w)')
        # for i in range(_mask.size(0)):
        #     index_per_batch = torch.where(_mask[i].bool())[0]
        #     indices.append(index_per_batch)
        mask_dict[(cur_h, cur_w)]['spatial']['indices'] = indices
        mask_dict[(cur_h, cur_w)]['spatial']['indices1'] = indices1
        mask_bool = _mask.bool()
        mask_bool = mask_bool.T
        device = mask.device
        batch_size, seq_len = mask_bool.shape
        # print('------------------')
        # time_stamp = time.time()
        arange_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # print('time for arange_indices:', time.time()-time_stamp)
        # time_stamp = time.time()
        nonzero_indices = torch.nonzero(mask_bool, as_tuple=True)
        valid_indices = torch.zeros_like(arange_indices)
        valid_indices[nonzero_indices[0], torch.cumsum(mask_bool.int(), dim=1)[mask_bool] - 1] = arange_indices[mask_bool]
        cumsum_mask = torch.cumsum(mask_bool.int(), dim=1)
        # print('time for cumsum_mask:', time.time()-time_stamp)
        # time_stamp = time.time()
        nearest_indices = torch.clip(cumsum_mask - 1, min=0)
        # print('time for nearest_indices:', time.time()-time_stamp)
        # time_stamp = time.time()
        actual_indices = valid_indices.gather(1, nearest_indices)
        mask_dict[(cur_h, cur_w)]['spatial']['actual_indices'] = actual_indices
        return mask_dict
        
    def compute_mask_dict_temporal(self, mask_dict, cur_h, cur_w):
        mask = mask_dict[(cur_h, cur_w)]['mask']
        indices = []
        _mask = torch.round(mask).to(torch.int)
        _mask = rearrange(_mask, 'b 1 t h w -> (b h w) (t)')
        indices1 = torch.nonzero(_mask.reshape(1, -1).squeeze(0))
        mask_dict[(cur_h, cur_w)]['temporal']['indices'] = indices
        mask_dict[(cur_h, cur_w)]['temporal']['indices1'] = indices1
        mask_bool = _mask.bool()
        device = mask.device
        batch_size, seq_len = mask_bool.shape
        arange_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        nonzero_indices = torch.nonzero(mask_bool, as_tuple=True)
        valid_indices = torch.zeros_like(arange_indices)
        valid_indices[nonzero_indices[0], torch.cumsum(mask_bool.int(), dim=1)[mask_bool] - 1] = arange_indices[mask_bool]
        cumsum_mask = torch.cumsum(mask_bool.int(), dim=1)
        nearest_indices = torch.clip(cumsum_mask - 1, min=0)
        actual_indices = valid_indices.gather(1, nearest_indices)
        mask_dict[(cur_h, cur_w)]['temporal']['actual_indices'] = actual_indices
        return mask_dict    
    
    def batched_find_idxs_to_keep(self, 
                            x: torch.Tensor, 
                            threshold: int=2, 
                            tubelet_size: int=2,
                            patch_size: int=16) -> torch.Tensor:
        """
        Find the static tokens in a video tensor, and return a mask
        that selects tokens that are not repeated.

        Args:
        - x (torch.Tensor): A tensor of shape [B, C, T, H, W].
        - threshold (int): The mean intensity threshold for considering
                a token as static.
        - tubelet_size (int): The temporal length of a token.
        Returns:
        - mask (torch.Tensor): A bool tensor of shape [B, T, H, W] 
            that selects tokens that are not repeated.

        """
        # Ensure input has the format [B, C, T, H, W]
        assert len(x.shape) == 5, "Input must be a 5D tensor"
        #ipdb.set_trace()
        # Convert to float32 if not already
        x = x.type(torch.float32)
        
        # Calculate differences between frames with a step of tubelet_size, ensuring batch dimension is preserved
        # Compare "front" of first token to "back" of second token
        diffs = x[:, :, (2*tubelet_size-1)::tubelet_size] - x[:, :, :-tubelet_size:tubelet_size]
        # Ensure nwe track negative movement.
        diffs = torch.abs(diffs)
        
        # Apply average pooling over spatial dimensions while keeping the batch dimension intact
        avg_pool_blocks = F.avg_pool3d(diffs, (1, patch_size, patch_size))
        # Compute the mean along the channel dimension, preserving the batch dimension
        avg_pool_blocks = torch.mean(avg_pool_blocks, dim=1, keepdim=True)
        # Create a dummy first frame for each item in the batch
        first_frame = torch.ones_like(avg_pool_blocks[:, :, 0:1]) * 255
        # first_frame = torch.zeros_like(avg_pool_blocks[:, :, 0:1])
        # Concatenate the dummy first frame with the rest of the frames, preserving the batch dimension
        avg_pool_blocks = torch.cat([first_frame, avg_pool_blocks], dim=2)
        # Determine indices to keep based on the threshold, ensuring the operation is applied across the batch
        # Update mask: 0 for high similarity, 1 for low similarity
        keep_idxs = avg_pool_blocks.squeeze(1) > threshold  
        keep_idxs = keep_idxs.unsqueeze(1)
        keep_idxs = keep_idxs.float()
        # Flatten out everything but the batch dimension
        # keep_idxs = keep_idxs.flatten(1)
        #ipdb.set_trace()
        return keep_idxs

    def compute_similarity_mask(self, latent, threshold=0.95):
        """
        Compute frame-wise similarity for latent and generate mask.

        Args:
        - latent (torch.Tensor): Latent tensor of shape [n, c, t, h, w].
        - threshold (float): Similarity threshold to determine whether to skip computation.

        Returns:
        - mask (torch.Tensor): Mask tensor of shape [n, 1, t, h, w],
        where mask = 0 means skip computation, mask = 1 means recompute.
        """
        n, c, t, h, w = latent.shape
        mask = torch.ones((n, 1, t, h, w), device=latent.device)  # Initialize mask with all 1s

        for frame_idx in range(1, t):  # Start from the second frame
            curr_frame = latent[:, :, frame_idx, :, :]  # Current frame [n, c, h, w]
            prev_frame = latent[:, :, frame_idx - 1, :, :]  # Previous frame [n, c, h, w]

            # Compute token-wise cosine similarity
            dot_product = (curr_frame * prev_frame).sum(dim=1, keepdim=True)  # [n, 1, h, w]
            norm_curr = curr_frame.norm(dim=1, keepdim=True)
            norm_prev = prev_frame.norm(dim=1, keepdim=True)
            similarity = dot_product / (norm_curr * norm_prev + 1e-8)  # Avoid division by zero

            # Update mask: 0 for high similarity, 1 for low similarity
            mask[:, :, frame_idx, :, :] = (similarity <= threshold).float()
        # mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        return mask
    
    def resize_mask(self, mask, target_h, target_w):
        """
        Resize the mask to match the new spatial dimensions of x.

        Args:
        - mask (torch.Tensor): Input mask of shape [b, 1, t, h, w].
        - target_h (int): Target height.
        - target_w (int): Target width.

        Returns:
        - resized_mask (torch.Tensor): Resized mask of shape [b, 1, t, target_h, target_w].
        """
        if mask is None:
            return mask
        batch, _, t, h, w = mask.shape

        if h == target_h and w == target_w:
            return mask  # No resizing needed

        # Reshape to [b * t, 1, h, w]
        mask = mask.view(batch * t, 1, h, w)

        # Resize to [b * t, 1, target_h, target_w]
        resized_mask = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Ensure the mask is binary (0 or 1)
        resized_mask = (resized_mask > 0.5).float()

        # Reshape back to [b, 1, t, target_h, target_w]
        resized_mask = resized_mask.view(batch, 1, t, target_h, target_w)

        return resized_mask
    @classmethod
    def from_pretrained_2d(cls, pretrained_model_name_or_path, unet_additional_kwargs={}, **kwargs):
        from diffusers import __version__
        from diffusers.utils import DIFFUSERS_CACHE, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, is_safetensors_available
        from diffusers.modeling_utils import load_state_dict
        print(f"loaded 3D unet's pretrained weights from {pretrained_model_name_or_path} ...")

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        model_file = None
        if is_safetensors_available():
            try:
                model_file = cls._get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=SAFETENSORS_WEIGHTS_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
            except:
                pass

        if model_file is None:
            model_file = cls._get_model_file(
                pretrained_model_name_or_path,
                weights_name=WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )

        config, unused_kwargs = cls.load_config(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            **kwargs,
        )

        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        model = cls.from_config(config, **unused_kwargs, **unet_additional_kwargs)
        state_dict = load_state_dict(model_file)

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        params = [p.numel() if "motion_modules." in n else 0 for n, p in model.named_parameters()]
        print(f"### Motion Module Parameters: {sum(params) / 1e6} M")
        
        return model

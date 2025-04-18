from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from ..layers import VanillaSelfAttention, VanillaCrossAttention

from x_transformers.attend import Attend
from x_transformers import (
    RMSNorm,
    FeedForward
)

from .hyper_connections import (
    HyperConnections,
    Residual
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash = False,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: tuple[Tensor],
        masks: tuple[Tensor | None] | None = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

# class
class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_text,
        dim_motion,
        dim_music,
        dim_cond=None,
        dim_head=64,
        heads=8,
        qk_rmsnorm=False,
        flash_attn=False,
        num_residual_streams=1,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        
        latent_dim = dim_motion
        dropout = 0.1

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.text_attn_residual_fn = residual_klass(num_residual_streams, dim=dim_text)
        self.text_ff_residual_fn = residual_klass(num_residual_streams, dim=dim_text)

        self.motion_attn_residual_fn = residual_klass(num_residual_streams, dim=dim_motion)
        self.motion_ff_residual_fn = residual_klass(num_residual_streams, dim=dim_motion)

        self.music_attn_residual_fn = residual_klass(num_residual_streams, dim=dim_music)
        self.music_ff_residual_fn = residual_klass(num_residual_streams, dim=dim_music)
                
        self.motion_self_attn = VanillaSelfAttention(latent_dim, heads, dropout)
        self.motion_norm1 = nn.LayerNorm(latent_dim)
        self.cross_attn = VanillaCrossAttention(latent_dim, latent_dim, heads, dropout, latent_dim)
        self.motion_norm2 = nn.LayerNorm(latent_dim)
        
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            dim_gammas = (
                *((dim_text,) * 4),
                *((dim_motion,) * 4),
                *((dim_music,) * 4)
            )

            dim_betas = (
                *((dim_text,) * 2),
                *((dim_motion,) * 2),
                *((dim_music,) * 2)
            )

            self.cond_dims = (*dim_gammas, *dim_betas)

            to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                to_cond_linear
            )

            nn.init.zeros_(to_cond_linear.weight)
            nn.init.zeros_(to_cond_linear.bias)
            nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine=not has_cond)
        self.motion_attn_layernorm = nn.LayerNorm(dim_motion, elementwise_affine=not has_cond)
        self.music_attn_layernorm = nn.LayerNorm(dim_music, elementwise_affine=not has_cond)

        self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine=not has_cond)
        self.motion_ff_layernorm = nn.LayerNorm(dim_motion, elementwise_affine=not has_cond)
        self.music_ff_layernorm = nn.LayerNorm(dim_music, elementwise_affine=not has_cond)

        self.joint_attn = JointAttention(
            dim_inputs=(dim_text, dim_motion, dim_music),
            dim_head=dim_head,
            heads=heads,
            flash=flash_attn
        )

        self.text_ff = FeedForward(dim_text, **ff_kwargs)
        self.motion_ff = FeedForward(dim_motion, **ff_kwargs)
        self.music_ff = FeedForward(dim_music, **ff_kwargs)

    def forward(
        self,
        *,
        text_tokens,
        motion_tokens,
        music_tokens,
        text_mask=None,
        time_cond=None,
        motion_cond=None,
        skip_feedforward_text_tokens=True
    ):
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        if self.has_cond:
            (
                text_pre_attn_gamma,
                text_post_attn_gamma,
                text_pre_ff_gamma,
                text_post_ff_gamma,
                motion_pre_attn_gamma,
                motion_post_attn_gamma,
                motion_pre_ff_gamma,
                motion_post_ff_gamma,
                music_pre_attn_gamma,
                music_post_attn_gamma,
                music_pre_ff_gamma,
                music_post_ff_gamma,
                text_pre_attn_beta,
                text_pre_ff_beta,
                motion_pre_attn_beta,
                motion_pre_ff_beta,
                music_pre_attn_beta,
                music_pre_ff_beta,
            ) = self.to_cond(time_cond).split(self.cond_dims, dim=-1)

        #Add motion cross attention
        if exists(motion_cond):
            motion_cond = self.motion_norm1(motion_cond)
            motion_cond = self.motion_self_attn(motion_cond)
            motion_cond = self.motion_norm2(motion_cond)
            motion_tokens = self.cross_attn(motion_tokens, motion_cond)
        
        
        text_tokens, add_text_residual = self.text_attn_residual_fn(text_tokens)
        motion_tokens, add_motion_residual = self.motion_attn_residual_fn(motion_tokens)
        music_tokens, add_music_residual = self.music_attn_residual_fn(music_tokens)

        text_tokens = self.text_attn_layernorm(text_tokens)
        motion_tokens = self.motion_attn_layernorm(motion_tokens)
        music_tokens = self.music_attn_layernorm(music_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
            motion_tokens = motion_tokens * motion_pre_attn_gamma + motion_pre_attn_beta
            music_tokens = music_tokens * music_pre_attn_gamma + music_pre_attn_beta

        text_tokens, motion_tokens, music_tokens = self.joint_attn(
            inputs=(text_tokens, motion_tokens, music_tokens),
            masks=(text_mask, None, None)
        )

        if self.has_cond:
            text_tokens = text_tokens * text_post_attn_gamma
            motion_tokens = motion_tokens * motion_post_attn_gamma
            music_tokens = music_tokens * music_post_attn_gamma

        text_tokens = add_text_residual(text_tokens)
        motion_tokens = add_motion_residual(motion_tokens)
        music_tokens = add_music_residual(music_tokens)

        if not skip_feedforward_text_tokens:
            text_tokens, add_text_residual = self.text_ff_residual_fn(text_tokens)
            text_tokens = self.text_ff_layernorm(text_tokens)
            if self.has_cond:
                text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta

        motion_tokens, add_motion_residual = self.motion_ff_residual_fn(motion_tokens)
        motion_tokens = self.motion_ff_layernorm(motion_tokens)
        if self.has_cond:
            motion_tokens = motion_tokens * motion_pre_ff_gamma + motion_pre_ff_beta

        music_tokens, add_music_residual = self.music_ff_residual_fn(music_tokens)
        music_tokens = self.music_ff_layernorm(music_tokens)
        if self.has_cond:
            music_tokens = music_tokens * music_pre_ff_gamma + music_pre_ff_beta

        motion_tokens = self.motion_ff(motion_tokens)
        if self.has_cond:
            motion_tokens = motion_tokens * motion_post_ff_gamma
        motion_tokens = add_motion_residual(motion_tokens)

        music_tokens = self.music_ff(music_tokens)
        if self.has_cond:
            music_tokens = music_tokens * music_post_ff_gamma
        music_tokens = add_music_residual(music_tokens)

        if skip_feedforward_text_tokens:
            return text_tokens, motion_tokens, music_tokens

        text_tokens = self.text_ff(text_tokens)
        if self.has_cond:
            text_tokens = text_tokens * text_post_ff_gamma
        text_tokens = add_text_residual(text_tokens)

        return text_tokens, motion_tokens, music_tokens

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_motion,
        dim_music,
        num_register_tokens=0,
        final_norm=True,
        num_residual_streams=4,
        **block_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable=num_residual_streams == 1)

        self.has_register_tokens = num_register_tokens > 0
        self.register_tokens = nn.Parameter(torch.zeros(num_register_tokens, dim_motion))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.blocks = ModuleList([])
        for _ in range(depth):
            block = MMDiTBlock(
                dim_motion=dim_motion,
                dim_music=dim_music,
                num_residual_streams=num_residual_streams,
                **block_kwargs
            )
            self.blocks.append(block)

        self.norm = RMSNorm(dim_motion) if final_norm else nn.Identity()

    def forward(
        self,
        *,
        text_tokens,
        motion_tokens,
        music_tokens,
        motion_cond,
        text_mask=None,
        time_cond=None,
        should_skip_last_feedforward=True
    ):
        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=motion_tokens.shape[0])
            motion_tokens, packed_shape = pack([register_tokens, motion_tokens], 'b * d')

        text_tokens = self.expand_streams(text_tokens)
        motion_tokens = self.expand_streams(motion_tokens)
        music_tokens = self.expand_streams(music_tokens)
        motion_cond = self.expand_streams(motion_cond)

        for ind, block in enumerate(self.blocks):
            is_last = ind == (len(self.blocks) - 1)
            text_tokens, motion_tokens, music_tokens = block(
                time_cond=time_cond,
                motion_cond=motion_cond,
                text_tokens=text_tokens,
                motion_tokens=motion_tokens,
                music_tokens=music_tokens,
                text_mask=text_mask,
                skip_feedforward_text_tokens=is_last and should_skip_last_feedforward
            )

        if self.has_register_tokens:
            _, motion_tokens = unpack(motion_tokens, packed_shape, 'b * d')

        text_tokens = self.reduce_streams(text_tokens)
        motion_tokens = self.reduce_streams(motion_tokens)
        music_tokens = self.reduce_streams(music_tokens)

        motion_tokens = self.norm(motion_tokens)

        return text_tokens, motion_tokens, music_tokens
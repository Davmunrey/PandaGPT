import copy
import fnmatch
import logging
from functools import partial
from typing import Callable, List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, trunc_normal_


class Attention(nn.Module):
    """
    Attention mechanism for the transformer model.
    """
    ...


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron used as a feed-forward network in the transformer.
    """
    ...


class MultiheadAttention(nn.MultiheadAttention):
    """
    Multi-head attention mechanism.
    """
    ...


class ViTAttention(Attention):
    """
    ViT Attention mechanism. Compatible with the Vision Transformer.
    """
    ...


class BlockWithMasking(nn.Module):
    """
    A block of the transformer that includes masking.
    """
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self._forward_with_optional_scaling(x, attn_mask)

    def _forward_with_optional_scaling(self, x: torch.Tensor, attn_mask: torch.Tensor):
        if self.layer_scale_type is None:
            return self._forward_without_scaling(x, attn_mask)
        else:
            return self._forward_with_scaling(x, attn_mask)

    def _forward_without_scaling(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
        return x + self.drop_path(self.mlp(self.norm_2(x)))

    def _forward_with_scaling(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x = (
            x
            + self.drop_path(self.attn(self.norm_1(x), attn_mask))
            * self.layer_scale_gamma1
        )
        return x + self.drop_path(self.mlp(self.norm_2(x))) * self.layer_scale_gamma2


_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class SimpleTransformer(nn.Module):
    """
    Simple Transformer with several features such as masked attention, DropPath, LayerScale, and Dropout in Attention and FFN.
    """
    def __init__(self, ...):
        self.drop_path_rate = self._get_drop_path_rate(drop_path_rate, drop_path_type, num_blocks)
        ...

    def forward(self, ...):
        return self._forward_with_optional_checkpointing(tokens, attn_mask, use_checkpoint, checkpoint_every_n, checkpoint_blk_ids)

    def _get_drop_path_rate(self, drop_path_rate, drop_path_type, num_blocks):
        if drop_path_type == "progressive":
            return [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            return [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

    def _forward_with_optional_checkpointing(self, tokens, attn_mask, use_checkpoint, checkpoint_every_n, checkpoint_blk_ids):
        if use_checkpoint and checkpoint_blk_ids:
            # use checkpointing only for specified blocks
            for i, blk in enumerate(self.blocks):
                if i in checkpoint_blk_ids:
                    tokens = checkpoint.checkpoint(blk, tokens, attn_mask)
                else:
                    tokens = blk(tokens, attn_mask)
        elif use_checkpoint:
            # use checkpointing for all blocks with a certain frequency
            for i, blk in enumerate(self.blocks):
                if i % checkpoint_every_n == 0:
                    tokens = checkpoint.checkpoint(blk, tokens, attn_mask)
                else:
                    tokens = blk(tokens, attn_mask)
        else:
            # do not use checkpointing
            for blk in self.blocks:
                tokens = blk(tokens, attn_mask)
        return self.head(tokens)

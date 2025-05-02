# coding=utf-8
# Copyright 2025 Ugoboss Inc. team. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
"JAX/Flax Falcon3 single-chip model for text generation." 

import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
import numpy as np

from flax.linen.partitioning import param_with_axes, with_sharding_constraint
# from jax.experimental.mesh_utils import Mesh
from jax.sharding import Mesh as ShardingMesh
from jax.sharding import PartitionSpec as P
from configuration_falcon import FalconConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel

class DenseLayer(nn.Module):
    """Linear layer for Falcon model."""
    in_features: int
    out_features: int
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = nn.initializers.normal(stddev=0.02)
    bias_init: Callable = nn.initializers.normal(stddev=0.02)
    
    def setup(self):
        self.weight = self.param('kernel', self.kernel_init, (self.out_features, self.in_features))
        if self.use_bias:
            self.bias = self.param('bias', self.bias_init, (self.out_features,))
        else:
            self.bias = None
        
    def __call__(self, x):
        #print(f"dense: x: {x.shape}, weight: {self.weight.shape}")
        out = jnp.matmul(x, self.weight.T)
        if self.use_bias:
            out += self.bias
        return out
    
def rotate_by_quarter(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate the last dimension of x by pi/4 (90 degrees).
    
    Args:
        x (`jnp.ndarray`):
            Input tensor of shape [batch_size, seq_len, dim]
            or [batch_size, seq_len, num_heads, head_dim]
    Returns:
        Rotated tensor of shape [batch_size, seq_len, dim]
        or [batch_size, seq_len, num_heads, head_dim]
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    unsqueeze_dim: int = 1
) -> jnp.ndarray:
    """
    Apply rotary position embedding to x.
    
    Args:
        q (`jnp.ndarray`):
            Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k (`jnp.ndarray`):
            Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        cos (`jnp.ndarray`):
            Cosine tensor of shape [batch_size, seq_len, hidden_size]
        sin (`jnp.ndarray`):
            Sine tensor of shape [batch_size, seq_len, hidden_size]
        unsqueeze_dim (`int`):
            Dimension to unsqueeze for cos and sin tensors
    Returns:
        Tuple of q and k tensors after applying rotary position embedding
    """
    sin = jnp.expand_dims(sin, unsqueeze_dim)
    cos = jnp.expand_dims(cos, unsqueeze_dim)
    q_embed = q * cos + rotate_by_quarter(q) * sin
    k_embed = k * cos + rotate_by_quarter(k) * sin
    return q_embed, k_embed

class RotaryPositionEmbedding(nn.Module):
    """Rotary position embedding for Falcon model."""
    config: FalconConfig

    def setup(self):
        self.max_seq_len_cached = self.config.max_position_embeddings
        self.attention_scaling = 1.0  # Default scaling
        self.dtype = self.config.dtype
        # setup inv_freq for rotary embeddings
        partial_rotary_factor = self.config.partial_rotary_factor if hasattr(self.config, "partial_rotary_factor") else 1.0
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (self.config.rope_theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.int32) / self.dim))

    def __call__(self, x: jnp.array, position_ids: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """
        Args:S
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            position_ids: Position IDs of shape [batch_size, seq_len]
        Returns:
            Tuple of cos and sin tensors for rotary embeddings
        """
        # [1, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].astype(x.dtype)
        # [batch_size, dim/2, 1]
        inv_freq_expanded = jnp.broadcast_to(inv_freq_expanded, (position_ids.shape[0], inv_freq_expanded.shape[1], 1))
        # [batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].astype(x.dtype)
        # [.., dim/2, 1] @ [..., 1, seq_len] = [batch_size, dim/2, seq_len]
        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded)
        # [batch_size, seq_len, dim/2]
        freqs = jnp.transpose(freqs, (0, 2, 1))
        # [batch_size, seq_len, dim]
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)

def dropout_add(
    x: jnp.ndarray,
    residual: jnp.ndarray,
    dropout_prob: float = 0.0,
    training: bool = False
) -> jnp.ndarray:
    """
    Applies dropout and adds residual.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, hidden_size]
        residual: Residual tensor of shape [batch_size, seq_len, hidden_size]
        dropout_prob: Dropout probability
        training: Boolean indicating whether in training mode
    Returns:
        Tensor of shape [batch_size, seq_len, hidden_size] after dropout and residual addition
    
    """
    out = nn.Dropout(rate=dropout_prob)(x, deterministic=not training)
    out = out + residual
    return out

def split_heads(fused_qkv: jnp.ndarray, config: FalconConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Split the last dimension into (num_heads, head_dim),
    while output shares same memory as `fused_qkv`.
    
    Args:
        qkv(`jnp.ndarray`):
            Input tensor of shape [batch_size, seq_len, qkv_out_dim]
        config(`FalconConfig`):
            Configuration object for Falcon model              
    
    Returns:
        query('jnp.ndarray'):
            Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        key('jnp.ndarray'):
            Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        value('jnp.ndarray'):
            Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, _ = fused_qkv.shape
    if config.group_query:
        qkv = jnp.reshape(fused_qkv, (batch_size, seq_len, -1, config.num_heads // config.num_kv_heads + 2, config.head_dim))
        query = qkv[..., :-2, :]
        key = qkv[..., [-2], :]
        value = qkv[..., [-1], :]
        print(f"query shape: {query.shape}, key shape: {key.shape}")
        key = jnp.broadcast_to(key, query.shape)
        value = jnp.broadcast_to(value, query.shape)
        query, key, value = [jnp.reshape(x,(batch_size, seq_len, -1, config.head_dim)) for x in (query, key, value)]
        print(f"query shape: {query.shape}, key shape: {key.shape}")
        return query, key, value
    elif config.multi_query:
        qkv = jnp.reshape(batch_size, seq_len, config.num_heads + 2, config.head_dim)
        return qkv[..., :-2, :], qkv[..., [-2], :], qkv[..., [-1], :]
    else:
        qkv = jnp.reshape(fused_qkv, (batch_size, seq_len, config.num_heads, 3, config.head_dim))
        return qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]

def merge_heads(x: jnp.ndarray, config: FalconConfig) -> jnp.ndarray:
    """
    Merge head together over the last dimension.

    Args:
        x(`jnp.ndarray`):
            Input tensor of shape [batch_size * num_heads, seq_len, head_dim]
        config(`FalconConfig`):
            Configuration object for Falcon model
    
    Returns:
        Merged tensor of shape [batch_size, seq_len, num_heads * head_dim]
    """
    # We want to achieve:
    # [batch_size * num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads * head_dim]
    num_heads = config.num_attention_heads
    batch_size_and_num_heads, seq_len, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads
    # [batch_size, num_heads, seq_len, head_dim]
    x = x.reshape(batch_size, num_heads, seq_len, config.head_dim)
    # [batch_size, seq_len, num_heads, head_dim]
    x = x.transpose(0, 2, 1, 3)
    # [batch_size, seq_len, num_heads * head_dim]
    x = x.reshape(batch_size, seq_len, -1)
    return x

class AttentionLayer(nn.Module):
    """Multi-head Attention layer for Falcon model."""
    config: FalconConfig
    is_causal: bool = True
 
    def setup(self, layer_idx = None):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.hidden_dropout = self.config.hidden_dropout
        self.max_position_embeddings = self.config.max_position_embeddings
        self.new_decoder_architecture = self.config.new_decoder_architecture
        if self.config.group_query:
            self.num_kv_heads = self.config.num_kv_heads if self.config.num_kv_heads is not None else 1
        elif self.config.multi_query:
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = self.num_heads
        if layer_idx is not None:
            self.layer_idx = layer_idx
        else:
            print("Layer index not provided, could indicate to bugs in the forward pass because caching is used.")
        
        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        
        # Determine output dimension for QKV projection
        if self.config.group_query:
            qkv_out_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        elif self.config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        
        # Set up rotary position embedding
        self.rope = RotaryPositionEmbedding(self.config)
        # Create QKV projection
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, qkv_out_dim]
        self.query_key_value = DenseLayer(
            in_features=self.hidden_size,
            out_features=qkv_out_dim,
            use_bias=self.config.bias,
            dtype=self.config.dtype
        )
        # Create output projection
        # [..., hidden_size] -> [..., hidden_size]
        self.dense = DenseLayer(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            use_bias=self.config.bias,
            dtype=self.config.dtype
        )
        self.attention_dropout = nn.Dropout(rate=self.config.attention_dropout)

    
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        kv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        output_attentions: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            hidden_states (`jnp.ndarray`):
                Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (`jnp.ndarray`):
                Attention mask of shape [batch_size, seq_len]
            position_ids (`jnp.ndarray`):
                Position IDs of shape [batch_size, seq_len]
            kv_cache (`Dict[str, jnp.ndarray]`, *optional*):
                Key-Value cache for past key-value pairs
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # [batch_size, seq_len, qkv_out_dim]
        fused_qkv = self.query_key_value(hidden_states)
        (query, key, value) = split_heads(fused_qkv, self.config)
        # [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len, _, _ = query.shape
        # [batch_size, num_heads, seq_len, head_dim]
        (query, key, value) = [jnp.transpose(x, (0, 2, 1, 3)) for x in (query, key, value)]
        query = jnp.reshape(query, (batch_size, self.num_heads, seq_len, self.head_dim))
        key = jnp.reshape(key, (batch_size, self.num_heads, seq_len, self.head_dim))
        value = jnp.reshape(value, (batch_size, self.num_heads, seq_len, self.head_dim))

        cos, sin = self.rope(hidden_states, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)

        if kv_cache is not None:
            # Use cached key and value if available
            key = jnp.concatenate([kv_cache["key"], key], axis=2)
            value = jnp.concatenate([kv_cache["value"], value], axis=2)
            kv_cache["key"] = key
            kv_cache["value"] = value
        
        # [batch_size, num_heads, seq_len, head_dim] @
        # [batch_size, num_heads, head_dim, seq_len] =
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = jnp.matmul(query, key.transpose(0, 1, 3, 2)) 
        attention_scores *= self.inv_norm_factor
        #print(f"Attention shape {attention_scores.shape}")

        if attention_mask is not None:
            # [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]
            # [batch_size, num_heads, seq_len, seq_len]
            attention_mask = jnp.broadcast_to(attention_mask, (batch_size, self.num_heads, seq_len, seq_len))
        
        # Apply attention mask
        attention_scores = nn.softmax(attention_scores + attention_mask, axis=-1)
        attention_output = jnp.matmul(attention_scores, value)

        # [batch_size, num_heads, seq_len, head_dim]
        attention_output = jnp.reshape(attention_output, (batch_size, self.num_heads, seq_len, self.head_dim))
        # [batch_size, seq_len, num_heads, head_dim]
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        # [batch_size, seq_len, hidden_size]
        attention_output = jnp.reshape(attention_output, (batch_size, seq_len, self.hidden_size))

        attention_output = self.dense(attention_output)

        if output_attentions:
            return attention_output, kv_cache, attention_scores
        else:
            return attention_output, kv_cache
        
class MLPBlock(nn.Module):
    """Feed-forward layer for Falcon model."""
    config: FalconConfig

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.layer_norm_epsilon = self.config.layer_norm_epsilon
        self.hidden_dropout = self.config.hidden_dropout
        
        # First dense layer
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, ffn_hidden_size]
        self.dense_h_to_4h = DenseLayer(
            in_features=self.hidden_size,
            out_features=self.ffn_hidden_size,
            use_bias=self.config.bias,
            dtype=self.config.dtype
        )
        
        # Second dense layer
        # [batch_size, seq_len, ffn_hidden_size] -> [batch_size, seq_len, hidden_size]
        self.dense_4h_to_h = DenseLayer(
            in_features=self.ffn_hidden_size,
            out_features=self.hidden_size,
            use_bias=self.config.bias,
            dtype=self.config.dtype
        )

        # Dropout layer if needed
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        self.dropout = nn.Dropout(rate=self.hidden_dropout)

        if self.config.activation == "gelu":
            self.activation = nn.gelu
        else:
            self.activation = nn.relu

    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x (`jnp.ndarray`):
                Input tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        x = self.dense_h_to_4h(x)
        x = self.activation(x)
        x = self.dense_4h_to_h(x)

        return x

class DecoderLayer(nn.Module):
    """Decoder layer for Falcon model."""
    config: FalconConfig
    layer_idx: Optional[int] = None

    def setup(self):
        self.attention = AttentionLayer(self.config, self.layer_idx)
        self.mlp = MLPBlock(self.config)
        self.dropout = dropout_add
        self.hidden_dropout = self.config.hidden_dropout

        # Layer normalization if using sequential attention
        # input -> layernorm -> attention ->
        # -> ?(hidden dropout/add) -> layernorm -> mlp -> output
        if not self.config.parallel_attn:
            self.config.num_ln_in_parallel_attn = 1
            self.input_layernorm = nn.LayerNorm(
                self.config.layer_norm_epsilon,
                dtype=self.config.dtype
            )
            self.post_attn_layernorm = nn.LayerNorm(
                self.config.layer_norm_epsilon,
                dtype=self.config.dtype
            )
        # Layer normalization if using parallel attention
        else:
            # Default to 2 if not specified
            if self.config.num_ln_in_parallel_attn is None:
                self.config.num_ln_in_parallel_attn = 2
            
            #                      /-> attention -\
            # input -> layernorm -<                >-> output
            #                      \-------> mlp -/
            if self.config.num_ln_in_parallel_attn == 1:
                self.input_layernorm = nn.LayerNorm(
                    self.config.layer_norm_epsilon,
                    dtype=self.config.dtype
                )
            #         /-> layernorm -> attention -\
            # input -<                             >-> output
            #         \-> layernorm -------> mlp -/
            else:
                self.attn_layernorm = nn.LayerNorm(
                    self.config.layer_norm_epsilon,
                    dtype=self.config.dtype
                )
                self.mlp_layernorm = nn.LayerNorm(
                    self.config.layer_norm_epsilon,
                    dtype=self.config.dtype
                )

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> jnp.ndarray:
        """
        Args:
            hidden_states (`jnp.ndarray`):
                Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (`jnp.ndarray`):
                Attention mask of shape [batch_size, seq_len]
            position_ids (`jnp.ndarray`):
                Position IDs of shape [batch_size, seq_len]
            use_cache (`bool`):
                Whether to use cache for key-value pairs.
            output_attentions (`bool`):
                Whether to output attention scores.
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states

        # Layer normalization before attention
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        if self.config.parallel_attn and self.config.num_ln_in_parallel_attn == 2:
            attn_layernorm_out = self.attn_layernorm(hidden_states)
            mlp_layernorm_out = self.mlp_layernorm(hidden_states)
        else:
            attn_layernorm_out = self.input_layernorm(hidden_states)

        # Attention layer
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        attn_outputs = self.attention(
            attn_layernorm_out,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=None if not use_cache else {},
            output_attentions=output_attentions
        )
        attention_output = attn_outputs[0]

        if not self.config.parallel_attn:
            # Residual connection and dropout
            residual = self.dropout(
                attention_output,
                residual,
                dropout_prob=self.hidden_dropout,
            )
            mlp_layernorm_out = self.post_attn_layernorm(residual)
        elif self.config.num_ln_in_parallel_attn == 1:
            mlp_layernorm_out = attention_output

        # MLP layer
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.parallel_attn:
            mlp_output += attention_output

        output = self.dropout(
            mlp_output,
            residual,
            dropout_prob=self.hidden_dropout,
        )
        if output_attentions:
            # return hidden_states, past_kv, attentions
            return (output,) + attn_outputs[1:] 
        else:
            # return hidden_states, past_kv
            return (output, attn_outputs[1])
        
class FalconPretrainedModel(FlaxPreTrainedModel):
    """Base class for all Falcon models."""
    config_class = FalconConfig
    base_model_prefix = "falcon"
    supports_gradient_checkpointing = True

    def __init__(self, config: FalconConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
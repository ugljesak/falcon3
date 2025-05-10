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

import jax
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
from output_models import *
from transformers.cache_utils import Cache
from utils import KVCache 

class DenseLayer(nn.Module):
    """Linear layer for Falcon model."""
    config: FalconConfig
    in_features: int
    out_features: int
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self, kernel_init: Callable = None, bias_init: Callable = None):
        if kernel_init is None:
            kernel_init = nn.initializers.normal(stddev=self.config.initializer_range)
        if bias_init is None:
            bias_init = nn.initializers.zeros
        self.weight = self.param('kernel', kernel_init, (self.out_features, self.in_features))
        if self.use_bias:
            self.bias = self.param('bias', bias_init, (self.out_features,))
        else:
            self.bias = None
        
    def __call__(self, x):
        #print(f"dense: x: {x.shape}, weight: {self.weight.shape}")
        out = jnp.matmul(x, self.weight.T)
        if self.use_bias:
            out += self.bias
        return out
    
def rotate_by_quarter(x: jax.Array) -> jax.Array:
    """
    Rotate the last dimension of x by pi/4 (90 degrees).
    
    Args:
        x (`jax.Array`):
            Input tensor of shape [batch_size, seq_len, dim]
            or [batch_size, seq_len, num_heads, head_dim]
    Returns:
        Rotated tensor of shape [batch_size, seq_len, dim]
        or [batch_size, seq_len, num_heads, head_dim]
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int = 1
) -> jax.Array:
    """
    Apply rotary position embedding to x.
    
    Args:
        q (`jax.Array`):
            Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k (`jax.Array`):
            Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        cos (`jax.Array`):
            Cosine tensor of shape [batch_size, seq_len, hidden_size]
        sin (`jax.Array`):
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

    def __call__(self, x: jax.Array, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Args:S
            x(`jax.Array`): 
                Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            position_ids (`jax.Array`):
                Position IDs of shape [batch_size, seq_len]
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
    x: jax.Array,
    residual: jax.Array,
    dropout_prob: float = 0.0,
    training: bool = False
) -> jax.Array:
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

def split_heads(fused_qkv: jax.Array, config: FalconConfig) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Split the last dimension into (num_heads, head_dim),
    while output shares same memory as `fused_qkv`.
    
    Args:
        qkv(`jax.Array`):
            Input tensor of shape [batch_size, seq_len, qkv_out_dim]
        config(`FalconConfig`):
            Configuration object for Falcon model              
    
    Returns:
        query('jax.Array'):
            Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        key('jax.Array'):
            Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        value('jax.Array'):
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
            config=self.config,
            in_features=self.hidden_size,
            out_features=qkv_out_dim,
            use_bias=self.config.bias,
            dtype=self.config.dtype
        )
        # Create output projection
        # [..., hidden_size] -> [..., hidden_size]
        self.dense = DenseLayer(
            config=self.config,
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            use_bias=self.config.bias,
            dtype=self.config.dtype
        )
        self.attention_dropout = nn.Dropout(
            rate=self.config.attention_dropout,
            deterministic=False
        )

    
    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        use_cache: bool = True,
        kv_cache: Optional[KVCache] = None,
        head_mask: Optional[jax.Array] = None,
        cache_position: Optional[jax.Array] = None,
        position_embeddings: Optional[Tuple[jax.Array, jax.Array]] = None,
        output_attentions: bool = False
    ) -> Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
        """
        Args:
            hidden_states (`jax.Array`):
                Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (`jax.Array`):
                Attention mask of shape [batch_size, 1, seq_len, max_seq_len]
            position_ids (`jax.Array`):
                Position IDs of shape [batch_size, seq_len]
            use_cache (`bool`):
                Whether to use key-value cache for faster decoding
            kv_cache (`Dict[str, jax.Array]`, *optional*):
                Key-Value cache for past key-value pairs
            head_mask (`jax.Array`, *optional*):
                Mask for attention heads of size [batch_size, num_heads, seq_len, seq_len]
            cache_position (`jax.Array`, *optional*):
                Position IDs for cache
            position_embeddings (`Tuple[jax.Array, jax.Array]`, *optional*):
                Cos and sin tensors for rotary embeddings
            output_attentions (`bool`):
                Whether to return attention scores
        
        Returns:
            Output tuple consisting of:
                Tensor of shape [batch_size, seq_len, hidden_size]
                Optional key-value cache for past key-value pairs
                Optional attention scores of shape [batch_size, num_heads, seq_len, seq_len]
        """
        print(f"AttentionLayer: hidden_states shape: {hidden_states.shape}, attention_mask shape: {attention_mask.shape}, position_ids shape: {position_ids.shape}")
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

        if position_embeddings is None:
            cos, sin = self.rope(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
        
        # Use cached key and value if available
        if use_cache:
            if kv_cache is None:
                kv_cache = KVCache(self.config, batch_size, key, value)
            else:
                key, value = kv_cache.update(key, value, cache_position, cos, sin)
        
        # [batch_size, num_heads, seq_len, head_dim] @
        # [batch_size, num_heads, head_dim, seq_len] =
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = jnp.matmul(query, key.transpose(0, 1, 3, 2)) 
        attention_scores *= self.inv_norm_factor
        #print(f"Attention shape {attention_scores.shape}")

        # Attention mask: [batch_size, 1, seq_len, max_seq_len]
        if attention_mask is not None:
            # [batch_size, 1, seq_len, seq_len]
            print(f"Attention mask: {attention_mask.shape}, key shape: {key.shape}")
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
            print(f"Attention mask after slicing: {attention_mask.shape}")
            # [batch_size, num_heads, seq_len, seq_len]
            attention_mask = jnp.broadcast_to(attention_mask, (batch_size, self.num_heads, seq_len, seq_len))
        
        # Apply attention mask
        attention_scores = nn.softmax(attention_scores + attention_mask, axis=-1)
        # Apply dropout if needed
        attention_scores = self.attention_dropout(attention_scores)
        # Apply head_mask
        if head_mask is not None:
            attention_scores = attention_scores * head_mask

        # [batch_size, num_heads, seq_len, seq_len] @
        # [batch_size, num_heads, seq_len, head_dim] =
        # [batch_size, num_heads, seq_len, head_dim]
        attention_output = jnp.matmul(attention_scores, value)
        # [batch_size, seq_len, num_heads, head_dim]
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        # [batch_size, seq_len, hidden_size]
        attention_output = jnp.reshape(attention_output, (batch_size, seq_len, self.hidden_size))

        # Apply final dense layer
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        attention_output = self.dense(attention_output)

        outputs = (attention_output,)
        if use_cache:
            outputs += (kv_cache,)
        if output_attentions:
            outputs += (attention_scores,)

        return outputs
        
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
            config=self.config,
            in_features=self.hidden_size,
            out_features=self.ffn_hidden_size,
            use_bias=self.config.bias,
            dtype=self.config.dtype
        )
        
        # Second dense layer
        # [batch_size, seq_len, ffn_hidden_size] -> [batch_size, seq_len, hidden_size]
        self.dense_4h_to_h = DenseLayer(
            config=self.config,
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

    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x (`jax.Array`):
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
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        head_mask: Optional[jax.Array] = None,
        use_cache: bool = False,
        kv_cache: Optional[KVCache] = None,
        cache_position: Optional[jax.Array] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[jax.Array, jax.Array]] = None,
    ) -> jax.Array:
        """
        Args:
            hidden_states (`jax.Array`):
                Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (`jax.Array`):
                Attention mask of shape [batch_size, seq_len]
            position_ids (`jax.Array`):
                Position IDs of shape [batch_size, seq_len]
            head_mask (`jax.Array`, *optional*):
                Mask for attention heads of size [batch_size, num_heads, seq_len, seq_len]
            use_cache (`bool`):
                Whether to use cache for key-value pairs.
            kv_cache (`Dict[str, jax.Array]`, *optional*):
                Key-Value cache for past key-value pairs
            cache_position (`jax.Array`, *optional*):
                Position IDs for cache
            output_attentions (`bool`):
                Whether to output attention scores.
            position_embeddings (`Tuple[jax.Array, jax.Array]`, *optional*):
                Cos and sin tensors for rotary embeddings

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
        print(f"attn_layernorm_out shape: {attn_layernorm_out.shape}")
        attn_outputs = self.attention(
            attn_layernorm_out,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            kv_cache=kv_cache if use_cache else None,
            cache_position=cache_position,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings
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
        
class FalconPreTrainedModel(nn.Module):
    """Derivative for falcon pretrained model."""
    config: FalconConfig 
    base_model_prefix: str = "falcon"

    def __init__(self):
        pass
        
    def _init_weights(self, module: nn.Module):
        """Initialize the weights of the model."""
        if isinstance(module, DenseLayer) or isinstance(module, nn.Dense):
            module.kernel_init = nn.initializers.normal(stddev=self.config.initializer_range)
            module.bias_init = nn.initializers.zeros
        elif isinstance(module, nn.LayerNorm):
            module.bias_init = nn.initializers.zeros
            module.weight_init = nn.initializers.ones
        elif isinstance(module, nn.Embed):
            module.embedding_init = nn.initializers.normal(stddev=self.config.initializer_range)
        else:
            raise ValueError(f"Unknown module type: {type(module)}")

def configure_head_mask(head_mask : Optional[jax.Array], num_hidden_layers: int) -> jax.Array:
    """
    Prepares the head mask for the model.

    Args:
        head_mask (`jax.Array`):
            Mask for attention heads of shape [num_heads] or [num_layers, num_heads]
        num_hidden_layers (`int`):
            Number of hidden layers in the model

    Returns:
        head_mask (`jax.Array`):
            Mask for attention heads of shape [num_layers, batch_size, num_heads, seq_len, seq_len]
    """
    if head_mask is None:
        return [None] * num_hidden_layers

    if head_mask.ndim == 1:
        # [num_heads] -> [1, 1, num_heads, 1, 1]
        head_mask = head_mask[None, None, :, None, None]
        print(f"head_mask shape: {head_mask.shape}")
        # [1, 1, num_heads, 1, 1] -> [num_layers, 1, num_heads, 1, 1]
        head_mask = jnp.broadcast_to(head_mask, (num_hidden_layers, 1, head_mask.shape[2], 1, 1))
    
    elif head_mask.ndim == 2:
        # [num_layers, num_heads] -> [num_layers, 1, num_heads, 1, 1]
        head_mask = head_mask[:, None, :, None, None]
        
    return head_mask

def update_causal_mask(
    config: FalconConfig,
    attention_mask: jax.Array,
    cache_position: jax.Array,
):
    """
    Update the causal mask for the model.

    Args:
        attention_mask (`jax.Array`):
            Attention mask of shape [batch_size, seq_len] or [batch_size, 1, query_lenght, kv_length]
        cache_position (`jax.Array`):
            Position IDs for cache
        past_key_values (`KVCache`):
            Key-Value cache for past key-value pairs

    Returns:
        Updated attention mask of shape [batch_size, 1, 1, seq_len]
    """
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask

    # Attention mask: [batch_size, seq_len]
    target_length = config.max_position_embeddings
    batch_size, seq_len = attention_mask.shape
    min_dtype = jnp.finfo(config.dtype).min
    print(f"min_dtype: {min_dtype}, attention_mask shape: {attention_mask.shape}")
    causal_mask = jnp.full((seq_len, target_length), fill_value=min_dtype, dtype=config.dtype)
    print(f"causal_mask shape: {causal_mask.shape}")
    if seq_len != 1:
        causal_mask = jnp.triu(causal_mask, k=1)
    print(f"cache_position shape: {cache_position.shape}")
    causal_mask *= jnp.arange(target_length) > cache_position[:, None]
    causal_mask = causal_mask[None, None, :, :].repeat(batch_size, axis=0)
    #causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, target_length))
    
    if attention_mask is not None:
        mask_length = attention_mask.shape[-1]
        
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        
        causal_mask = causal_mask.at[:, :, :, :mask_length].set(
            jnp.where(padding_mask, min_dtype, causal_mask[:, :, :, :mask_length])
        )

    return causal_mask


class FalconModel(FalconPreTrainedModel):
    """Base class for all Falcon models."""
    config: FalconConfig

    def setup(self):

        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        
        # Set up the embedding layer to transform input tokens into embeddings
        self.word_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.embed_dim,
            embedding_init=nn.initializers.xavier_uniform(),
            dtype=self.config.dtype
        )

        # Set up the transformer blocks
        self.blocks = [DecoderLayer(self.config, layer_idx=i) for i in range(self.config.num_hidden_layers)]

        # Set up final layer normalization
        self.final_layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.config.dtype
        )

        self.rotary_emb = RotaryPositionEmbedding(config=self.config)
        

    def get_input_embeddings(self):
        return self.word_embeddings
    
    def set_input_embeddings(self, new_embeddings: jax.Array):
        self.word_embeddings = new_embeddings

    def __call__(
        self,
        input_ids: Optional[jax.Array] = None,
        input_embeds: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        head_mask: Optional[jax.Array] = None,
        kv_cache: Optional[Dict[str, jax.Array]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[jax.Array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Args:
            input_ids (`jax.Array`):
                Input tensor of shape [batch_size, seq_len]
            input_embeds (`jax.Array`):
                Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (`jax.Array`):
                Attention mask of shape [batch_size, seq_len]
            position_ids (`jax.Array`):
                Position IDs of shape [batch_size, seq_len]
            head_mask (`jax.Array`):
                Mask for attention heads of shape [num_heads] or [num_layers, num_heads]
            kv_cache (`Dict[str, jax.Array]`, *optional*):
                Key-Value cache for past key-value pairs
            use_cache (`bool`):
                Whether to use cache for key-value pairs.
            cache_position (`jax.Array`, *optional*):
                Position IDs for cache
            output_attentions (`bool`):
                Whether to output attention scores.
            output_hidden_states (`bool`):
                Whether to output hidden states.
            return_dict (`bool`):
                Whether to return a dictionary or a tuple.
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError("You have to specify exactly one of input_ids or input_embeds.")
        
        print(f"Input ids: {input_ids.shape}")
        if input_embeds is None:
            input_embeds = self.word_embeddings(input_ids)
        print(f"input embeds: {input_embeds.shape}")
        past_kv_len = kv_cache["key"].shape[-2] if kv_cache is not None else 0
        if cache_position is None:
            cache_position = jnp.arange(past_kv_len, past_kv_len + input_embeds.shape[1], dtype=jnp.int32)
        
        if position_ids is None:
            position_ids = cache_position[None, :]

        causal_mask = update_causal_mask(
            self.config,
            attention_mask=attention_mask,
            cache_position=cache_position
        )

        hidden_states = input_embeds
        head_mask = configure_head_mask(head_mask, self.config.num_hidden_layers)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            print(f"Block {i} input shape: {hidden_states.shape}")
            # Apply attention block
            outputs = block(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                kv_cache=kv_cache,
                cache_position=cache_position,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                position_embeddings=position_embeddings
            )

            # Unpack outputs
            hidden_states = outputs[0]
            # Update cache if needed
            if use_cache:
                next_decoder_cache = outputs[1]
            # Store attention scores if needed for every layer
            if output_attentions:
                all_self_attentions += (outputs[2],)

        # Apply final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return Tuple( v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attentions] 
                if v is not None)
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions
            )
        
class FalconForCausalLM(FalconPreTrainedModel):
    """Falcon model for causal language modeling."""

    def __init__(self, config: FalconConfig, *args, **kwargs):
        super.__init__(config, *args, **kwargs)
        
        # Set up the model
        self.transformer = FalconModel(config)
        
        # Set up the lm head
        self.lm_head = DenseLayer(
            config=config,
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            use_bias=False,
            dtype=config.dtype
        )

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings: jax.Array):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        head_mask: Optional[jax.Array] = None,
        kv_cache: Optional[Dict[str, jax.Array]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[jax.Array] = None,
        logits_to_keep: Union[int, jax.Array] = 0,
        **kwargs
    ) -> Union[Tuple[jax.Array], CausalLMOutputWithCrossAttentions]:
        """
        Args:
            All from FalconModel forward method.
            
            labels (`jax.Array`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
            
            logits_to_keep (`int`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `jax.Array`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:
            Output tensor of shape [batch_size, seq_len, vocab_size]
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Call the transformer model
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        lm_logits = self.lm_head(hidden_states[:, slice_indices, :])

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
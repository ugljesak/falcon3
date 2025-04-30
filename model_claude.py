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
import jax
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

# Constants for tensor parallelism
TP_AXIS_NAME = 'tp'
DP_AXIS_NAME = 'dp'

from .configuration_falcon import FalconConfig 

# Activation functions
def gelu_impl(x):
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def get_activation(activation_str):
    if activation_str == "gelu":
        return gelu_impl
    elif activation_str == "relu":
        return nn.relu
    else:
        return nn.gelu


# Rotary Position Embedding functions
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Alibi tensor function
def build_alibi_tensor(attention_mask, num_heads, dtype):
    """Build the ALiBi tensor for Falcon model."""
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** int(math.log2(num_heads))
    
    base = jnp.array(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=jnp.float32
    )
    powers = jnp.arange(1, 1 + closest_power_of_2, dtype=jnp.int32)
    slopes = jnp.power(base, powers)
    
    if closest_power_of_2 != num_heads:
        extra_base = jnp.array(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=jnp.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = jnp.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=jnp.int32)
        slopes = jnp.concatenate([slopes, jnp.power(extra_base, extra_powers)], axis=0)
    
    # Create arange tensor
    arange_tensor = ((jnp.cumsum(attention_mask, axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).astype(dtype)


class FalconLinear(nn.Module):
    """Linear layer for Falcon model."""
    in_features: int
    out_features: int
    bias: bool = False
    dtype: jnp.dtype = jnp.float16
    kernel_init: Callable = nn.initializers.normal(stddev=0.02)
    bias_init: Callable = nn.initializers.zeros
    tp_mesh: bool = True  # Whether to use tensor parallelism
    
    def setup(self):
        # Use param_with_axes to enable tensor parallelism 
        # We partition the weight along the output dim for tensor parallelism
        if self.tp_mesh:
            self.weight = param_with_axes(
                "weight",
                self.kernel_init,
                (self.in_features, self.out_features),
                jnp.float32,
                axes=('embed', 'mlp') if self.in_features != self.out_features else ('embed', 'embed')
            )
        else:
            self.weight = self.param(
                "weight", 
                self.kernel_init, 
                (self.in_features, self.out_features)
            )
            
        if self.bias:
            if self.tp_mesh:
                self.bias_param = param_with_axes(
                    "bias", 
                    self.bias_init, 
                    (self.out_features,), 
                    jnp.float32,
                    axes=('mlp',) if self.in_features != self.out_features else ('embed',)
                )
            else:
                self.bias_param = self.param(
                    "bias", 
                    self.bias_init, 
                    (self.out_features,)
                )
                
    def __call__(self, x):
        x = x @ self.weight
        if self.bias:
            x = x + self.bias_param
        return x.astype(self.dtype)


class FalconRotaryEmbedding(nn.Module):
    """Rotary position embeddings for Falcon."""
    config: FalconConfig
    
    def setup(self):
        self.max_seq_len_cached = self.config.max_position_embeddings
        self.dim = self.config.head_dim
        
        # Create inv_freq
        inv_freq = 1.0 / (self.config.rope_theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.inv_freq = inv_freq
        self.attention_scaling = 1.0  # Default scaling
        
    def __call__(self, x, position_ids):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            position_ids: Position IDs of shape [batch_size, seq_len]
            
        Returns:
            Tuple of cos and sin tensors for rotary embeddings
        """
        inv_freq_expanded = jnp.expand_dims(self.inv_freq, axis=(0, 2))
        position_ids_expanded = jnp.expand_dims(position_ids.astype(jnp.float32), axis=-1)
        
        # Compute freqs
        freqs = jnp.matmul(position_ids_expanded, inv_freq_expanded)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling
        
        return cos.astype(x.dtype), sin.astype(x.dtype)


class FalconAttention(nn.Module):
    """Multi-head attention for Falcon model with tensor parallelism."""
    config: FalconConfig
    layer_idx: Optional[int] = None
    is_causal: bool = True
    
    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1
        
        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        
        # Determine output dimension for QKV projection
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
            
        # Create QKV and output projections
        self.query_key_value = FalconLinear(
            in_features=self.hidden_size,
            out_features=qkv_out_dim,
            bias=config.bias,
            dtype=config.dtype
        )
        
        self.dense = FalconLinear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=config.bias,
            dtype=config.dtype
        )
        
        self.dropout = nn.Dropout(rate=config.attention_dropout)
        
        if config.rotary:
            self.rotary_emb = FalconRotaryEmbedding(config=config)
            
    def _split_heads(self, fused_qkv):
        """Split QKV into separate Q, K, V tensors."""
        batch_size, seq_length = fused_qkv.shape[:2]
        
        if self.new_decoder_architecture:
            # Handle new decoder architecture
            qkv = fused_qkv.reshape(batch_size, seq_length, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            # Broadcast key and value to match query shape
            key = jnp.broadcast_to(key, query.shape)
            value = jnp.broadcast_to(value, query.shape)
            
            query, key, value = [x.reshape(batch_size, seq_length, -1, self.head_dim) for x in (query, key, value)]
        elif not self.multi_query:
            # Standard multi-head attention
            fused_qkv = fused_qkv.reshape(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            query, key, value = [fused_qkv[:, :, :, i, :] for i in range(3)]
        else:
            # Multi-query attention
            fused_qkv = fused_qkv.reshape(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            query = fused_qkv[:, :, :-2, :]
            key = fused_qkv[:, :, [-2], :]
            value = fused_qkv[:, :, [-1], :]
            # Broadcast key and value for multi-query attention
            key = jnp.broadcast_to(key, query.shape)
            value = jnp.broadcast_to(value, query.shape)
            
        return query, key, value
    
    def _merge_heads(self, x):
        """Merge attention heads."""
        batch_size, seq_length, num_heads, head_dim = x.shape
        x = x.reshape(batch_size, seq_length, num_heads * head_dim)
        return x
        
    def __call__(
        self,
        hidden_states,
        alibi=None,
        attention_mask=None,
        position_ids=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        deterministic=True,
        output_attentions=False,
        cache_position=None,
        position_embeddings=None,
    ):
        batch_size, query_length, _ = hidden_states.shape
        
        # Project hidden states to query, key, value
        fused_qkv = self.query_key_value(hidden_states)
        query_layer, key_layer, value_layer = self._split_heads(fused_qkv)
        
        # Prepare query, key, value for attention
        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        query_layer = jnp.transpose(query_layer, (0, 2, 1, 3))
        key_layer = jnp.transpose(key_layer, (0, 2, 1, 3))
        value_layer = jnp.transpose(value_layer, (0, 2, 1, 3))
        
        # Apply rotary position embeddings if needed
        if alibi is None and position_embeddings is not None:
            cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)
        
        # Handle cached key/value states if using_cache
        kv_length = key_layer.shape[2]
        if layer_past is not None:
            if isinstance(layer_past, tuple):
                past_key, past_value = layer_past
                key_layer = jnp.concatenate([past_key, key_layer], axis=2)
                value_layer = jnp.concatenate([past_value, value_layer], axis=2)
                kv_length = key_layer.shape[2]
        
        # Save key/value states for future use if needed
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None
            
        # Create attention mask if needed
        if attention_mask is not None:
            # Extend attention mask to match key length
            attention_mask = attention_mask[:, :, :, :kv_length]
        
        # Calculate attention
        if alibi is None:
            # Standard attention with or without rotary embeddings
            # [batch_size, num_heads, query_length, key_length]
            attn_weights = jnp.matmul(query_layer, jnp.transpose(key_layer, (0, 1, 3, 2)))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if present
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            # Get attention probabilities
            attention_probs = nn.softmax(attn_weights, axis=-1)
            attention_probs = self.dropout(attention_probs, deterministic=deterministic)
            
            # Apply attention to values
            attn_output = jnp.matmul(attention_probs, value_layer)
        else:
            # Attention with ALiBi
            matmul_result = jnp.matmul(query_layer, jnp.transpose(key_layer, (0, 1, 3, 2)))
            
            # alibi shape is [batch_size * num_heads, 1, seq_length]
            # We need to reshape it to [batch_size, num_heads, 1, seq_length]
            alibi_reshaped = alibi.reshape(batch_size, self.num_heads, 1, -1)
            
            # Add ALiBi bias
            attention_scores = matmul_result + alibi_reshaped
            attention_scores = attention_scores * self.inv_norm_factor
            
            # Apply attention mask if present
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
                
            # Get attention probabilities
            attention_probs = nn.softmax(attention_scores, axis=-1)
            attention_probs = self.dropout(attention_probs, deterministic=deterministic)
            
            # Apply head mask if present
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
                
            # Apply attention to values
            attn_output = jnp.matmul(attention_probs, value_layer)
        
        # Reshape output
        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = self._merge_heads(attn_output)
        
        # Apply output projection
        attn_output = self.dense(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attention_probs,)
            
        return outputs


class FalconMLP(nn.Module):
    """MLP module for Falcon with tensor parallelism."""
    config: FalconConfig
    
    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        self.activation = get_activation(self.config.activation)
        
        # Create dense layers with tensor parallelism
        self.dense_h_to_4h = FalconLinear(
            in_features=self.hidden_size,
            out_features=self.ffn_hidden_size,
            bias=self.config.bias,
            dtype=self.config.dtype
        )
        
        self.dense_4h_to_h = FalconLinear(
            in_features=self.ffn_hidden_size,
            out_features=self.hidden_size,
            bias=self.config.bias,
            dtype=self.config.dtype
        )
        
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        
    def __call__(self, x, deterministic=True):
        x = self.dense_h_to_4h(x)
        x = self.activation(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class FalconDecoderLayer(nn.Module):
    """Transformer decoder layer for Falcon with tensor parallelism."""
    config: FalconConfig
    layer_idx: Optional[int] = None
    
    def setup(self):
        config = self.config
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # Create attention layer
        self.self_attention = FalconAttention(
            config=config,
            layer_idx=self.layer_idx
        )
        
        # Create MLP
        self.mlp = FalconMLP(config=config)
        
        # Set up layer normalization based on architecture
        if not config.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(
                epsilon=config.layer_norm_epsilon,
                dtype=config.dtype
            )
            self.input_layernorm = nn.LayerNorm(
                epsilon=config.layer_norm_epsilon,
                dtype=config.dtype
            )
        else:
            if config.num_ln_in_parallel_attn == 2:
                # The layer norm before self-attention
                self.ln_attn = nn.LayerNorm(
                    epsilon=config.layer_norm_epsilon,
                    dtype=config.dtype
                )
                # The layer norm before the MLP
                self.ln_mlp = nn.LayerNorm(
                    epsilon=config.layer_norm_epsilon,
                    dtype=config.dtype
                )
            else:
                self.input_layernorm = nn.LayerNorm(
                    epsilon=config.layer_norm_epsilon,
                    dtype=config.dtype
                )
    
    def __call__(
        self,
        hidden_states,
        alibi=None,
        attention_mask=None,
        position_ids=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        deterministic=True,
        output_attentions=False,
        cache_position=None,
        position_embeddings=None,
    ):
        residual = hidden_states
        
        # Determine which layernorm to use based on architecture
        if self.config.new_decoder_architecture and self.config.num_ln_in_parallel_attn == 2:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)
        
        # Self attention
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            alibi=alibi,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        
        # Handle different architectures
        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                # Add residual connection and apply dropout
                if not deterministic:
                    attention_output = nn.Dropout(rate=self.config.hidden_dropout)(attention_output)
                residual = residual + attention_output
                mlp_layernorm_out = self.post_attention_layernorm(residual)
        
        if (
            self.config.new_decoder_architecture
            and self.config.parallel_attn
            and self.config.num_ln_in_parallel_attn == 1
        ):
            mlp_layernorm_out = attention_layernorm_out
        
        # MLP
        mlp_output = self.mlp(mlp_layernorm_out, deterministic=deterministic)
        
        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output
        
        # Final dropout and residual connection
        if not deterministic:
            mlp_output = nn.Dropout(rate=self.config.hidden_dropout)(mlp_output)
        output = residual + mlp_output
        
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]
        
        return outputs


class FalconModel(nn.Module):
    """Falcon Transformer model with tensor parallelism."""
    config: FalconConfig
    
    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.use_alibi = self.config.alibi
        
        # Word embeddings
        self.word_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.config.dtype
        )
        
        # Transformer blocks
        self.h = [
            FalconDecoderLayer(config=self.config, layer_idx=i)
            for i in range(self.config.num_hidden_layers)
        ]
        
        # Final Layer Norm
        self.ln_f = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            dtype=self.config.dtype
        )
        
        # Rotary embeddings
        if self.config.rotary:
            self.rotary_emb = FalconRotaryEmbedding(config=self.config)
    
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic=True,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        
        # Compute position ids if not provided
        if position_ids is None:
            if past_key_values[0] is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
            else:
                past_key_values_length = 0
                
            if cache_position is None:
                cache_position = jnp.arange(past_key_values_length, past_key_values_length + seq_length)
                
            position_ids = jnp.expand_dims(cache_position, axis=0)
        
        # Compute alibi tensor for attention bias
        alibi = None
        if self.use_alibi:
            if attention_mask is None:
                attention_mask = jnp.ones((batch_size, seq_length + past_key_values_length), dtype=jnp.int32)
            alibi = build_alibi_tensor(attention_mask, self.num_heads, inputs_embeds.dtype)
        
        # Create causal mask for self attention
        causal_mask = None
        if attention_mask is not None:
            if past_key_values[0] is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
            else:
                past_key_values_length = 0
                
            # Create causal mask for auto-regressive decoding
            causal_mask = make_causal_mask(
                jnp.ones((batch_size, seq_length + past_key_values_length), dtype="bool"),
                dtype=inputs_embeds.dtype
            )
            
            if attention_mask is not None:
                # Combine with attention mask
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
                attention_mask = (1.0 - attention_mask) * jnp.finfo(inputs_embeds.dtype).min
                causal_mask = combine_masks(causal_mask, attention_mask)
        
        # Prepare head mask
        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = jnp.expand_dims(jnp.expand_dims(head_mask, axis=0), axis=0)
                head_mask = jnp.expand_dims(jnp.expand_dims(head_mask, axis=-1), axis=-1)
                head_mask = jnp.broadcast_to(head_mask, (self.config.num_hidden_layers, batch_size, self.num_heads, seq_length, seq_length))
            elif head_mask.ndim == 2:
                head_mask = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(head_mask, axis=-1), axis=-1), axis=0)
            else:
                raise ValueError(f"Head mask shape {head_mask.shape} not supported")
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        # Prepare position embeddings
        position_embeddings = None
        if not self.use_alibi and self.config.rotary:
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        # Run all decoder layers
        hidden_states = inputs_embeds
        
        next_decoder_cache = ()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = block(
                hidden_states,
                alibi=alibi,
                attention_mask=causal_mask,
                position_ids=position_ids,
                layer_past=layer_past,
                head_mask=None if head_mask is None else head_mask[i],
                use_cache=use_cache,
                deterministic=deterministic,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
                
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2 if use_cache else 1],)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_decoder_cache if use_cache else None,
                all_hidden_states,
                all_self_attentions
            ] if v is not None)
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache if use_cache else None,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }


class FalconForCausalLM(nn.Module):
    """Falcon Causal Language Model with tensor parallelism."""
    config: FalconConfig
    
    def setup(self):
        # Initialize the base transformer
        self.transformer = FalconModel(config=self.config)
        
        # LM head
        self.lm_head = FalconLinear(
            in_features=self.config.hidden_size,
            out_features=self.config.vocab_size,
            bias=False,
            dtype=self.config.dtype
        )
    
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic=True,
        cache_position=None,
        logits_to_keep=0,
    ):
        return_dict = return_dict if return_dict is not None else True
        
        # Run the transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            deterministic=deterministic,
            cache_position=cache_position,
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Compute logits only for specific tokens if specified
        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]
        else:  # Tensor case
            hidden_states = jnp.take(hidden_states, logits_to_keep, axis=1)
        
        # Apply language model head
        lm_logits = self.lm_head(hidden_states)
        
        if not return_dict:
            outputs = (lm_logits,)
            if use_cache:
                outputs = outputs + (transformer_outputs["past_key_values"],)
            if output_hidden_states:
                outputs = outputs + (transformer_outputs["hidden_states"],)
            if output_attentions:
                outputs = outputs + (transformer_outputs["attentions"],)
            return outputs
            
        return {
            "logits": lm_logits,
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
        }


# ------------------ Tensor Parallelism Implementation ------------------

def init_with_mesh(mesh, model, rng, input_shape):
    """Initialize model with a specific device mesh."""
    with mesh:
        return model.init(rng, *input_shape)


def partition_model_parameters(mesh, params):
    """Partition model parameters according to the mesh and PartitionSpec."""
    with mesh:
        return params


def get_jax_tensor_parallel_mesh(mesh_shape):
    """Create a JAX mesh for tensor parallelism.
    
    Args:
        mesh_shape: Tuple of (rows, cols) for the mesh.
        
    Returns:
        JAX mesh object for use with sharding.
    """
    devices = jax.devices()
    num_devices = len(devices)
    
    # Calculate total expected devices from mesh shape
    expected_devices = mesh_shape[0] * mesh_shape[1]
    
    if num_devices < expected_devices:
        raise ValueError(
            f"Not enough devices available. Requested mesh shape {mesh_shape} "
            f"requires {expected_devices} devices, but only {num_devices} available."
        )
    
    # Create device mesh
    device_mesh = np.array(devices[:expected_devices]).reshape(mesh_shape)
    
    # Create JAX mesh
    return (device_mesh, (DP_AXIS_NAME, TP_AXIS_NAME))


def apply_model_parallel(model_params, mesh, dp_axis_name, tp_axis_name):
    """Apply model parallelism to model parameters based on parameter patterns."""
    # Define rules for parameter sharding
    def get_partition_spec(path, leaf):
        path_str = "/".join(str(p) for p in path)
        
        # Embedding layer
        if "word_embeddings" in path_str:
            if "embedding" in path_str:
                return P(dp_axis_name, tp_axis_name)
            
        # Linear layer weights
        if "weight" in path_str:
            if any(x in path_str for x in ["query_key_value", "dense_h_to_4h"]):
                return P(None, tp_axis_name)
            if any(x in path_str for x in ["dense", "dense_4h_to_h", "lm_head"]):
                return P(tp_axis_name, None)
            
        # Biases
        if "bias" in path_str:
            if any(x in path_str for x in ["query_key_value", "dense_h_to_4h"]):
                return P(tp_axis_name)
            if any(x in path_str for x in ["dense", "dense_4h_to_h", "lm_head"]):
                return P(None)
            
        # Layer norm parameters - no sharding needed
        if any(x in path_str for x in ["ln_f", "input_layernorm", "post_attention_layernorm"]):
            return P(None)
            
        # Default: no sharding
        return P(None)
    
    # Apply sharding specs
    flat_params = flatten_dict(model_params)
    sharded_params = {}
    
    for path, param in flat_params.items():
        spec = get_partition_spec(path, param)
        sharded_params[path] = jax.lax.with_sharding_constraint(param, spec)
        
    return unflatten_dict(sharded_params)


# Load model weights from PyTorch checkpoint to JAX/Flax model
def load_pytorch_weights(config, pytorch_checkpoint_path):
    """
    Load weights from a PyTorch checkpoint into a JAX/Flax model.
    
    Args:
        config: FalconConfig instance
        pytorch_checkpoint_path: Path to PyTorch checkpoint
        
    Returns:
        Flax model parameters initialized with PyTorch weights
    """
    import torch
    from transformers import FalconForCausalLM as PTFalconForCausalLM
    
    # Load PyTorch model
    pt_model = PTFalconForCausalLM.from_pretrained(pytorch_checkpoint_path)
    pt_state_dict = pt_model.state_dict()
    
    # Initialize JAX model
    jax_model = FalconForCausalLM(config=config)
    rng = jax.random.PRNGKey(0)
    
    # Create sample input for initialization
    sample_input_ids = jnp.zeros((1, 16), dtype=jnp.int32)
    
    # Initialize JAX model parameters
    params = jax_model.init(rng, input_ids=sample_input_ids)
    
    # Map PyTorch parameters to JAX
    pt_to_jax_mapping = {
        # Embeddings
        "transformer.word_embeddings.weight": "transformer/word_embeddings/embedding",
        
        # Layer norms
        "transformer.ln_f.weight": "transformer/ln_f/scale",
        "transformer.ln_f.bias": "transformer/ln_f/bias",
        
        # LM head
        "lm_head.weight": "lm_head/weight",
    }
    
    # Add decoder layer mappings
    for i in range(config.num_hidden_layers):
        # Layer norms
        if not config.parallel_attn:
            pt_to_jax_mapping[f"transformer.h.{i}.input_layernorm.weight"] = f"transformer/h/{i}/input_layernorm/scale"
            pt_to_jax_mapping[f"transformer.h.{i}.input_layernorm.bias"] = f"transformer/h/{i}/input_layernorm/bias"
            pt_to_jax_mapping[f"transformer.h.{i}.post_attention_layernorm.weight"] = f"transformer/h/{i}/post_attention_layernorm/scale"
            pt_to_jax_mapping[f"transformer.h.{i}.post_attention_layernorm.bias"] = f"transformer/h/{i}/post_attention_layernorm/bias"
        else:
            if config.num_ln_in_parallel_attn == 2:
                pt_to_jax_mapping[f"transformer.h.{i}.ln_attn.weight"] = f"transformer/h/{i}/ln_attn/scale"
                pt_to_jax_mapping[f"transformer.h.{i}.ln_attn.bias"] = f"transformer/h/{i}/ln_attn/bias"
                pt_to_jax_mapping[f"transformer.h.{i}.ln_mlp.weight"] = f"transformer/h/{i}/ln_mlp/scale"
                pt_to_jax_mapping[f"transformer.h.{i}.ln_mlp.bias"] = f"transformer/h/{i}/ln_mlp/bias"
            else:
                pt_to_jax_mapping[f"transformer.h.{i}.input_layernorm.weight"] = f"transformer/h/{i}/input_layernorm/scale"
                pt_to_jax_mapping[f"transformer.h.{i}.input_layernorm.bias"] = f"transformer/h/{i}/input_layernorm/bias"
        
        # Self-attention
        pt_to_jax_mapping[f"transformer.h.{i}.self_attention.query_key_value.weight"] = f"transformer/h/{i}/self_attention/query_key_value/weight"
        if config.bias:
            pt_to_jax_mapping[f"transformer.h.{i}.self_attention.query_key_value.bias"] = f"transformer/h/{i}/self_attention/query_key_value/bias"
        pt_to_jax_mapping[f"transformer.h.{i}.self_attention.dense.weight"] = f"transformer/h/{i}/self_attention/dense/weight"
        if config.bias:
            pt_to_jax_mapping[f"transformer.h.{i}.self_attention.dense.bias"] = f"transformer/h/{i}/self_attention/dense/bias"
        
        # MLP
        pt_to_jax_mapping[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"] = f"transformer/h/{i}/mlp/dense_h_to_4h/weight"
        if config.bias:
            pt_to_jax_mapping[f"transformer.h.{i}.mlp.dense_h_to_4h.bias"] = f"transformer/h/{i}/mlp/dense_h_to_4h/bias"
        pt_to_jax_mapping[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"] = f"transformer/h/{i}/mlp/dense_4h_to_h/weight"
        if config.bias:
            pt_to_jax_mapping[f"transformer.h.{i}.mlp.dense_4h_to_h.bias"] = f"transformer/h/{i}/mlp/dense_4h_to_h/bias"
    
    # Convert params
    flax_params = unfreeze(params)
    
    for pt_key, jax_key in pt_to_jax_mapping.items():
        if pt_key in pt_state_dict:
            pt_tensor = pt_state_dict[pt_key].detach().numpy()
            
            # Navigate to the correct place in the params tree
            keys = jax_key.split('/')
            current = flax_params
            for i, k in enumerate(keys[:-1]):
                current = current[k]
            
            # Handle special cases (e.g., layer norms where PyTorch uses weight/bias but JAX uses scale/bias)
            current[keys[-1]] = pt_tensor
    
    return freeze(flax_params)


# Generate function with tensor parallelism
def generate(
    model,
    params,
    mesh,
    input_ids,
    max_length=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=None,
    eos_token_id=None,
):
    """
    Generate text with a tensor-parallelized Falcon model.
    
    Args:
        model: FalconForCausalLM model instance
        params: Model parameters
        mesh: JAX mesh for tensor parallelism
        input_ids: Input token ids
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        do_sample: Whether to sample or use greedy decoding
        pad_token_id: Pad token ID
        eos_token_id: End of sequence token ID
        
    Returns:
        Generated token ids
    """
    with mesh:
        # Apply model parallelism to parameters
        sharded_params = apply_model_parallel(
            params, mesh, DP_AXIS_NAME, TP_AXIS_NAME
        )
        
        # Prepare for generation
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        max_length = max_length if max_length is not None else 100
        
        # Setup token generation stopping criteria
        eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
        
        # Initialize past key values
        past_key_values = None
        
        # Cache for all generated tokens
        all_tokens = jnp.array(input_ids)
        
        # Generation function with JIT compilation for efficiency
        @jax.jit
        def sample_next_token(input_ids, past_key_values, rng):
            # Forward pass
            outputs = model.apply(
                sharded_params,
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                deterministic=True,
            )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Apply temperature
            logits = logits / jnp.maximum(temperature, 1e-7)
            
            # Apply top-k
            if top_k > 0:
                top_k_value, _ = jax.lax.top_k(logits, top_k)
                top_k_value = jnp.min(top_k_value, axis=-1, keepdims=True)
                logits = jnp.where(logits < top_k_value, -1e10, logits)
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
                sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
                cumulative_probs = jnp.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
                
                # Scatter sorted indices
                indices_to_remove = jnp.zeros_like(logits, dtype=bool)
                for batch_idx in range(batch_size):
                    indices_to_remove = indices_to_remove.at[batch_idx, sorted_indices[batch_idx]].set(
                        sorted_indices_to_remove[batch_idx]
                    )
                
                logits = jnp.where(indices_to_remove, -1e10, logits)
            
            # Sample from the filtered distribution
            if do_sample:
                next_token = jax.random.categorical(rng, logits, axis=-1)
            else:
                next_token = jnp.argmax(logits, axis=-1)
                
            return next_token, past_key_values
        
        # Generate tokens
        rng = jax.random.PRNGKey(0)
        for _ in range(max_length - cur_len):
            # For the first iteration, use the original input_ids
            # For subsequent iterations, use only the last generated token
            if past_key_values is None:
                current_input_ids = input_ids
            else:
                current_input_ids = all_tokens[:, -1:]
            
            # Split PRNG key
            rng, sample_rng = jax.random.split(rng)
            
            # Sample next token
            next_token, past_key_values = sample_next_token(
                current_input_ids, past_key_values, sample_rng
            )
            
            # Append to generated tokens
            all_tokens = jnp.concatenate([all_tokens, next_token[:, None]], axis=1)
            
            # Check if EOS token was generated
            if jnp.all(next_token == eos_token_id):
                break
        
        return all_tokens


# Example usage with tensor parallelism
def example_usage():
    """Example of how to use the tensor-parallelized Falcon model."""
    from transformers import AutoTokenizer
    
    # Configuration for Falcon3-7B
    config = FalconConfig(
        vocab_size=65024,
        hidden_size=3072,  # Adjusted for Falcon3-7B
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,  # For grouped-query attention
        max_position_embeddings=2048,
        rope_theta=10000.0,
        layer_norm_epsilon=1e-5,
        new_decoder_architecture=True,
        bias=False,
        alibi=False,
        rotary=True,
    )
    
    # Create tensor parallel mesh (2x4)
    mesh = get_jax_tensor_parallel_mesh((2, 4))
    
    # Initialize model with the mesh
    model = FalconForCausalLM(config=config)
    rng = jax.random.PRNGKey(0)
    
    # Load PyTorch weights (if available)
    # params = load_pytorch_weights(config, "path/to/falcon-model")
    
    # Or initialize from scratch
    sample_input = jnp.zeros((1, 16), dtype=jnp.int32)
    params = init_with_mesh(mesh, model, rng, (sample_input,))
    
    # Apply tensor parallelism to parameters
    sharded_params = apply_model_parallel(params, mesh, DP_AXIS_NAME, TP_AXIS_NAME)
    
    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    input_text = "Once upon a time, in a land far away,"
    input_ids = tokenizer.encode(input_text, return_tensors="np")
    input_ids = jnp.array(input_ids)
    
    # Generate text
    generated_ids = generate(
        model,
        sharded_params,
        mesh,
        input_ids,
        max_length=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tensor-parallelized Falcon model in JAX/Flax")
    parser.add_argument("--mesh_shape", type=str, default="2,4", help="Mesh shape for tensor parallelism (rows,cols)")
    parser.add_argument("--prompt", type=str, default="Once upon a time,", help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--model_path", type=str, default=None, help="Path to PyTorch model to convert")
    
    args = parser.parse_args()
    mesh_shape = tuple(map(int, args.mesh_shape.split(',')))
    
    example_usage()
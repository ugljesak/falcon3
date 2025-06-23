from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from .configuration_falcon3 import Falcon3Config
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from model.configuration_falcon3 import Falcon3Config
from transformers.models.llama.configuration_llama import LlamaConfig


def create_sinusoidal_positions(num_pos, theta, dim):
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype(jnp.float32)

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class FlaxFalcon3RMSNorm(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxFalcon3RotaryEmbedding(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, self.config.rope_theta, head_dim)

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query


class ShardedEmbedding(nn.Module):
    """Sharded embedding layer."""
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32
    tp_size: int = 1

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        # Shard vocabulary (rows) across TP dimension
        self.embedding = self.param(
            'embedding',
            embedding_init,
            (self.config.vocab_size, self.hidden_size),
            self.dtype
        )
        # Parameter will be sharded with PartitionSpec(None, 'tp')
        
    def __call__(self, input_ids):
        # Need to gather from all TP devices since vocabulary is sharded
        # Each device only has a subset of the embedding table
        input_ids = input_ids.astype('i4')
        
        # In actual implementation, we'll use a collective gather operation
        # For now, using a simple embedding lookup
        embedded = jnp.take(self.embedding, input_ids, axis=0)
        return embedded

class ShardedFalcon3Attention(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    
    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Determine heads per device for tensor parallelism
        self.tp_size = jax.process_count()  # Will be set properly during initialization
        self.heads_per_device = self.num_heads // self.tp_size
        self.kv_heads_per_device = self.num_key_value_heads // self.tp_size
        
        # Sharded projection matrices - each device only has a subset of heads
        # q_proj is sharded across TP dimension in its output dimension
        self.q_proj = nn.Dense(
            self.heads_per_device * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        
        # k_proj and v_proj are similarly sharded
        self.k_proj = nn.Dense(
            self.kv_heads_per_device * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        
        self.v_proj = nn.Dense(
            self.kv_heads_per_device * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        
        # o_proj is sharded in its input dimension
        self.o_proj = nn.Dense(
            self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        
        # Other components remain the same
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        self.rotary_emb = FlaxFalcon3RotaryEmbedding(config, dtype=self.dtype)
    
    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # Project to Q, K, V - each device only computes a subset of heads
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Split into heads
        query = self._split_heads(query, self.heads_per_device)
        key = self._split_heads(key, self.kv_heads_per_device)
        value = self._split_heads(value, self.kv_heads_per_device)
        
        # Apply rotary embeddings
        key, query = self.rotary_emb(key, query, position_ids)
        query_length, key_length = query.shape[1], key.shape[1]

        # Handle caching logic
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        if attention_mask is not None:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        else:
            attention_mask = causal_mask

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # Repeat KV heads to match Q heads if needed (group-query attention)
        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)
        
        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )
        # Compute attention scores and outputs - local to each device
        attention_dtype = jnp.float32 if self.dtype != jnp.float32 else self.dtype
        
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        )
        if attention_dtype != self.dtype:
            attn_weights = attn_weights.astype(self.dtype)
            
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        
        # All-reduce across TP dimension to get the full output
        attn_output = jax.lax.psum(attn_output, axis_name='tp')

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
    
class ShardedFalcon3MLP(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.act = nn.silu
        
        # Shard along the output dimension
        self.gate_proj = nn.Dense(
            inner_dim // self.tp_size,  # Each device computes a slice of activations
            use_bias=False, 
            dtype=self.dtype, 
            kernel_init=kernel_init
        )
        
        # Shard along the input dimension
        self.down_proj = nn.Dense(
            embed_dim, 
            use_bias=False, 
            dtype=self.dtype, 
            kernel_init=kernel_init
        )
        
        # Shard along the output dimension
        self.up_proj = nn.Dense(
            inner_dim // self.tp_size, 
            use_bias=False, 
            dtype=self.dtype, 
            kernel_init=kernel_init
        )

    def __call__(self, hidden_states):
        # Compute sharded projections
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)        
        # In actual implementation, use jax.lax.psum here for all-reduce
        return jax.lax.psum(hidden_states, axis_name='tp')
    
class ShardedFalcon3DecoderLayer(nn.Module):
    """Transformer decoder layer with tensor parallelism support."""
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32
    tp_size: int = 1  # Tensor parallel size
    
    def setup(self):
        self.input_layernorm = FlaxFalcon3RMSNorm(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxFalcon3RMSNorm(self.config, dtype=self.dtype)
        
        self.self_attn = ShardedFalcon3Attention(
            self.config,
            causal=True,
            dtype=self.dtype
        )
        self.mlp = ShardedFalcon3MLP(
            self.config,
            dtype=self.dtype
        )
        
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        
        hidden_states = attn_outputs[0]
        
        # All-reduce happens inside the attention module
        hidden_states = residual + hidden_states
        
        # Second residual connection
        residual = hidden_states
        
        # Second normalization - replicated across all devices
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP - computation sharded across TP dimension
        # All-reduce happens inside the MLP module
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection - happens locally on each device
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        
        # Optionally include attention weights in outputs
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
            
        return outputs

class ShardedFalcon3Model(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32
    tp_size: int = 1
    dp_size: int = 1
    
    def setup(self):
        self.hidden_size = self.config.hidden_size
        
        # Store mesh info
        self.tp_size = self.tp_size
        self.dp_size = self.dp_size
        
        # Create sharded components
        self.embed_tokens = ShardedEmbedding(self.config, dtype=self.dtype)
        self.layers = [
            ShardedFalcon3DecoderLayer(
                self.config, 
                dtype=self.dtype, 
                tp_size=self.tp_size,
                name=str(i)
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = FlaxFalcon3RMSNorm(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Embedding lookup (sharded across TP dimension)
        input_embeds = self.embed_tokens(input_ids)
        
        # Process through layers
        hidden_states = input_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        
        # Final normalization (replicated across all devices)
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Construct output dictionary/tuple
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
            
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }
    
class ShardedFalcon3ForCausalLM(nn.Module):
    """Falcon3 model with tensor-parallel and data-parallel capabilities for causal language modeling."""
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32
    tp_size: int = 1  # Tensor parallel size
    dp_size: int = 1  # Data parallel size
    
    def setup(self):
        self.model = ShardedFalcon3Model(
            config=self.config,
            dtype=self.dtype,
            tp_size=self.tp_size,
            dp_size=self.dp_size
        )
        
        # Language modeling head - sharded across TP dimension in input dimension
        # This matches the output dimension of the final transformer layer
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        
        # For storing caching information during generation
        self.cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Obtain transformer outputs
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the last hidden state from transformer outputs
        if return_dict:
            hidden_states = transformer_outputs['last_hidden_state']
        else:
            hidden_states = transformer_outputs[0]
        
        # Apply language modeling head - sharded across TP dimension
        # Need to all-gather across TP dimension before final projection
        # since each device only has a portion of hidden states
        
        # In practice, we'd use a collective operation here
        # For simplicity, assuming hidden_states is already gathered
        
        # Project to vocabulary size
        logits = self.lm_head(hidden_states)
        
        # Need to all-reduce logits across TP dimension as vocabulary is sharded
        # In practice, we'd use jax.lax.all_reduce here
        
        if not return_dict:
            outputs = (logits,) + transformer_outputs[1:]
            return outputs
            
        return {
            'logits': logits,
            'hidden_states': transformer_outputs.get('hidden_states', None),
            'attentions': transformer_outputs.get('attentions', None),
        }
    
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        """Prepare inputs for generation with proper sharding."""
        batch_size, seq_length = input_ids.shape
        
        # Initialize cache
        cache = self.init_cache(batch_size, max_length)
        
        # Initialize position IDs
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype="i4")
            
        position_ids = attention_mask.cumsum(axis=-1) - 1
        
        # Create extended attention mask for the max sequence length
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        
        return {
            "input_ids": input_ids,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
            "cache": cache,
        }
    
    def init_cache(self, batch_size, max_length):
        """Initialize KV cache with proper sharding."""
        config = self.config
        
        # Use head dimension from config
        head_dim = config.hidden_size // config.num_attention_heads
        
        # Create empty cache dict
        cache = {}
        cache['cache_index'] = jnp.zeros((), dtype=jnp.int32)
        
        # Adjust sharding for KV cache if using tensor parallelism
        # Each device only needs to store a subset of KV heads
        kv_heads_per_device = config.num_key_value_heads // self.tp_size
        
        # Initialize cache for each layer
        for i in range(config.num_hidden_layers):
            cache[f'layers.{i}.self_attn.cached_key'] = jnp.zeros(
                (batch_size, kv_heads_per_device, max_length, head_dim),
                dtype=jnp.float32
            )
            cache[f'layers.{i}.self_attn.cached_value'] = jnp.zeros(
                (batch_size, kv_heads_per_device, max_length, head_dim),
                dtype=jnp.float32
            )
        
        return freeze(cache)
    
    def get_sharded_params_dict(self, params):
        """Get parameter dictionary with proper sharding annotations."""
        # Flatten parameters
        flat_params = flatten_dict(params)
        
        # Create sharding rules
        rules = {}
        
        # Model embeddings
        rules['model/embed_tokens/embedding'] = P(None, 'tp')
        
        # For each layer
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f'model/layers/{i}/'
            
            # Attention module
            rules[f'{layer_prefix}self_attn/q_proj/kernel'] = P(None, 'tp')
            rules[f'{layer_prefix}self_attn/k_proj/kernel'] = P(None, 'tp')
            rules[f'{layer_prefix}self_attn/v_proj/kernel'] = P(None, 'tp')
            rules[f'{layer_prefix}self_attn/o_proj/kernel'] = P('tp', None)
            
            # MLP module
            rules[f'{layer_prefix}mlp/gate_proj/kernel'] = P(None, 'tp')
            rules[f'{layer_prefix}mlp/up_proj/kernel'] = P(None, 'tp')
            rules[f'{layer_prefix}mlp/down_proj/kernel'] = P('tp', None)
            
            # Layer norms (replicated)
            rules[f'{layer_prefix}input_layernorm/weight'] = P()
            rules[f'{layer_prefix}post_attention_layernorm/weight'] = P()
        
        # Final norm and LM head
        rules['model/norm/weight'] = P()
        rules['lm_head/kernel'] = P('tp', None)  # Output tied to embeddings
        
        # Apply sharding rules
        sharded_flat_params = {}
        for key, param in flat_params.items():
            param_name = '/'.join([str(k) for k in key])
            if param_name in rules:
                sharded_flat_params[key] = param, rules[param_name]
            else:
                sharded_flat_params[key] = param, P()  # Replicate by default
        
        # Reconstruct parameter tree
        return unflatten_dict(sharded_flat_params)
    
    def shard_model(self, params, mesh):
        """Apply sharding to model parameters using a device mesh."""
        # Get sharded parameter dictionary
        sharded_params_dict = self.get_sharded_params_dict(params)
        
        # Apply sharding
        sharded_params = {}
        for key, (param, spec) in sharded_params_dict.items():
            sharded_params[key] = jax.device_put(
                param, 
                jax.sharding.NamedSharding(mesh, spec)
            )
        
        return unflatten_dict(sharded_params)
    
    def generate(
        self,
        input_ids,
        attention_mask,
        position_ids,
        params,
        mesh,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """Generate text with tensor parallelism."""
        batch_size, seq_length = input_ids.shape
        
        # Set default token IDs if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        # Shard parameters
        sharded_params = self.shard_model(params, mesh)
        
        # Prepare inputs with proper caching
        inputs = self.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=seq_length + max_new_tokens,
            attention_mask=attention_mask,
        )
        
        # Initial forward pass with full input
        def forward_fn(params, input_ids, attention_mask, position_ids):
            outputs = self.apply(
                params,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )
            return outputs['logits']
        
        # Run initial forward pass with sharding
        with mesh:
            logits = jax.jit(
                forward_fn,
                in_shardings=(None, P(None, None), P(None, None), P(None, None)),
                out_shardings=P(None, None, None)  # (batch, seq, vocab)
            )(sharded_params, inputs['input_ids'], inputs['attention_mask'], inputs['position_ids'])
        
        # Get first generated token
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature and sampling
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            
        # Apply top-k if specified
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
            next_token_logits = jnp.zeros_like(next_token_logits).at[
                jnp.arange(batch_size)[:, None], top_k_indices
            ].set(top_k_logits)
        
        # Apply softmax for probabilities
        probs = jax.nn.softmax(next_token_logits, axis=-1)
        
        # Sample next token
        next_token = jax.random.categorical(
            jax.random.PRNGKey(0),  # Should use a proper RNG key
            next_token_logits,
            axis=-1
        )
        
        # Start collecting generated tokens
        all_tokens = [input_ids, next_token[:, None]]
        
        # Initialize generation variables
        has_reached_eos = jnp.zeros((batch_size,), dtype=bool)
        
        # Auto-regressive generation loop
        for i in range(1, max_new_tokens):
            # Update position IDs for next token
            current_position = seq_length + i
            position_id = jnp.array([[current_position - 1]])
            
            # Create attention mask for the next token
            attention_mask_step = jnp.ones((batch_size, 1), dtype=jnp.int32)
            
            # Single-token forward function
            def step_fn(params, next_token, position_id, attention_mask):
                outputs = self.apply(
                    params,
                    input_ids=next_token,
                    position_ids=position_id,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                return outputs['logits']
            
            # Run step with sharding
            with mesh:
                next_logits = jax.jit(
                    step_fn,
                    in_shardings=(None, P(None, None), P(None, None), P(None, None)),
                    out_shardings=P(None, None, None)
                )(sharded_params, next_token[:, None], position_id, attention_mask_step)
            
            # Get token logits for the next position
            next_token_logits = next_logits[:, -1, :]
            
            # Apply temperature and sampling as before
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-k if specified
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
                next_token_logits = jnp.zeros_like(next_token_logits).at[
                    jnp.arange(batch_size)[:, None], top_k_indices
                ].set(top_k_logits)
            
            # Apply softmax for probabilities
            probs = jax.nn.softmax(next_token_logits, axis=-1)
            
            # Sample next token
            next_token = jax.random.categorical(
                jax.random.PRNGKey(i),  # Should use a proper RNG key
                next_token_logits,
                axis=-1
            )
            
            # Apply EOS mask if needed
            if eos_token_id is not None:
                next_token = jnp.where(has_reached_eos, pad_token_id, next_token)
                has_reached_eos = has_reached_eos | (next_token == eos_token_id)
                
                # Early stopping if all sequences have reached EOS
                if jnp.all(has_reached_eos):
                    break
            
            # Add to generated tokens
            all_tokens.append(next_token[:, None])
        
        # Concatenate all generated tokens
        generated_ids = jnp.concatenate(all_tokens, axis=1)
        return generated_ids



def sharded_forward(model, params, input_ids, attention_mask, position_ids, mesh):
    """Forward pass with proper sharding and collectives."""
    
    # Shard inputs across DP dimension
    dp_size = mesh.shape['dp']
    batch_size = input_ids.shape[0]
    batch_size_per_device = batch_size // dp_size
    
    # This would be handled by the input pipeline in practice
    input_ids = input_ids.reshape(dp_size, batch_size_per_device, -1)
    attention_mask = attention_mask.reshape(dp_size, batch_size_per_device, -1)
    position_ids = position_ids.reshape(dp_size, batch_size_per_device, -1)
    
    def forward_fn(params, input_ids, attention_mask, position_ids):
        # Forward pass with model
        outputs = model.apply(
            params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=True,
            return_dict=True
        )
        
        # All-reduce logits across TP dimension if needed
        # This is handled internally in the sharded modules
        
        return outputs['logits']
    
    with mesh:
        # Shard inputs across DP dimension
        input_ids_sharded = jax.device_put(
            input_ids, 
            jax.sharding.NamedSharding(mesh, P('dp', None, None))
        )
        attention_mask_sharded = jax.device_put(
            attention_mask,
            jax.sharding.NamedSharding(mesh, P('dp', None, None))
        )
        position_ids_sharded = jax.device_put(
            position_ids,
            jax.sharding.NamedSharding(mesh, P('dp', None, None))
        )
        
        # JIT the forward function with proper annotations
        sharded_logits = jax.jit(
            forward_fn,
            in_shardings=(None, P('dp', None, None), P('dp', None, None), P('dp', None, None)),
            out_shardings=P('dp', None, None, None)
        )(params, input_ids_sharded, attention_mask_sharded, position_ids_sharded)
    
    return sharded_logits

def sharded_generate(
    model, 
    params, 
    input_ids, 
    attention_mask, 
    position_ids, 
    max_new_tokens,
    mesh
):
    """Generate text with a sharded model."""
    # This implementation simplifies by assuming batch_size=1 for generation
    # Real implementation would handle multiple sequences properly
    
    batch_size, seq_length = input_ids.shape
    
    with mesh:
        # Initialize KV cache
        cache = model.init_cache(batch_size, seq_length + max_new_tokens)
        
        # First forward pass with full input
        def initial_forward(params, input_ids, attention_mask, position_ids, cache):
            outputs, new_cache = model.apply(
                params,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                mutable=['cache'],
            )
            next_token = jnp.argmax(outputs['logits'][:, -1, :], axis=-1)
            return next_token, new_cache
        
        # Shard inputs appropriately
        input_ids_sharded = jax.device_put(
            input_ids,
            jax.sharding.NamedSharding(mesh, P(None, None))  # Replicated across devices
        )
        attention_mask_sharded = jax.device_put(
            attention_mask,
            jax.sharding.NamedSharding(mesh, P(None, None))
        )
        position_ids_sharded = jax.device_put(
            position_ids,
            jax.sharding.NamedSharding(mesh, P(None, None))
        )
        
        # Get first token and update cache
        next_token, cache = jax.jit(
            initial_forward,
            in_shardings=(None, P(None, None), P(None, None), P(None, None), None),
            out_shardings=(None, None)
        )(params, input_ids_sharded, attention_mask_sharded, position_ids_sharded, cache)
        
        all_tokens = [input_ids, next_token[:, None]]
        
        # Auto-regressive generation
        for i in range(1, max_new_tokens):
            current_position = seq_length + i
            
            # JIT the step function
            def generation_step(params, next_token, position_id, attention_mask, cache):
                outputs, new_cache = model.apply(
                    params,
                    input_ids=next_token,
                    position_ids=position_id,
                    attention_mask=attention_mask,
                    return_dict=True,
                    mutable=['cache'],
                )
                next_token = jnp.argmax(outputs['logits'][:, -1, :], axis=-1)
                return next_token, new_cache
            
            # Prepare inputs for this step
            position_id = jnp.array([[current_position - 1]])
            attention_mask_step = jnp.ones((1, 1))
            
            # Execute step
            next_token, cache = jax.jit(
                generation_step,
                in_shardings=(None, None, None, None, None),
                out_shardings=(None, None)
            )(params, next_token[:, None], position_id, attention_mask_step, cache)
            
            all_tokens.append(next_token[:, None])
            
            # Optional: Check for EOS
            
        # Concatenate all tokens
        generated_ids = jnp.concatenate(all_tokens, axis=1)
        return generated_ids
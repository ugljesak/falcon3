import jax
import numpy as np
from transformers.models.falcon.modeling_falcon import (
    FalconRotaryEmbedding,
    FalconDecoderLayer,
)
from model.model_falcon import MLPBlock, AttentionLayer, DecoderLayer
import jax.numpy as jnp
import torch
from .test_utils import compare_results
from model.configuration_falcon import FalconConfig

config = FalconConfig(group_query=True, num_kv_heads=1, num_attention_heads=30, hidden_size=480)

batch_size = 8
seq_len = 50
num_heads = config.num_attention_heads
head_dim = config.head_dim
hidden_size = config.hidden_size

x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
x_jax = jnp.array(x_np)
x_torch = torch.tensor(x_np)
attention_mask_jax = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
attention_mask_torch = torch.ones((batch_size, seq_len), dtype=torch.float32)
position_ids_jax = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
position_ids_torch = torch.Tensor(np.array(position_ids_jax)).to(torch.int32)

torch_rope = FalconRotaryEmbedding(config)
pos_embeddings = torch_rope(x_torch, position_ids_torch)
attention_mask_torch = attention_mask_torch[:, None, None, :]

decoder_jax = DecoderLayer(config=config)
decoder_torch = FalconDecoderLayer(config)

vars = decoder_jax.init(jax.random.PRNGKey(0), x_jax, attention_mask_jax, position_ids_jax)
vars['params']['attention']['query_key_value']['kernel'] = jnp.array(decoder_torch.self_attention.query_key_value.weight.detach().numpy())
vars['params']['attention']['dense']['kernel'] = jnp.array(decoder_torch.self_attention.dense.weight.detach().numpy())
vars['params']['mlp']['dense_h_to_4h']['kernel'] = jnp.array(decoder_torch.mlp.dense_h_to_4h.weight.detach().numpy())
vars['params']['mlp']['dense_4h_to_h']['kernel'] = jnp.array(decoder_torch.mlp.dense_4h_to_h.weight.detach().numpy())

decoder_out_jax, _, attn_out_jax = decoder_jax.apply(vars, x_jax, attention_mask=attention_mask_jax, position_ids=position_ids_jax, output_attentions=True)
decoder_out_torch, attn_out_torch = decoder_torch(x_torch, attention_mask=attention_mask_torch, position_ids=position_ids_torch, position_embeddings=pos_embeddings, output_attentions=True, alibi=None)
compare_results(decoder_out_jax, jnp.array(decoder_out_torch.detach().numpy()))
compare_results(attn_out_jax, jnp.array(attn_out_torch.detach().numpy()))
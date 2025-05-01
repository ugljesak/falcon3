import jax
import numpy as np
from transformers.models.falcon.modeling_falcon import (
    FalconLinear,
    FalconAttention,
    FalconRotaryEmbedding
)
from transformers.models.falcon.configuration_falcon import (
    FalconConfig, 
)
from model import merge_heads, split_heads, AttentionLayer 
import jax.numpy as jnp
import torch
from utils import compare_results
from configuration_falcon import FalconConfig

config = FalconConfig(group_query=True, num_kv_heads=1, num_attention_heads=30, hidden_size=480)
batch_size = 8
seq_len = 50
num_heads = config.num_attention_heads
head_dim = config.head_dim
hidden_dim = config.hidden_size
torch_attn = FalconAttention(config)
num_qkv_heads = (num_heads + 2 * config.num_kv_heads)

np.random.seed(0)
x_np = np.random.randn(batch_size, seq_len, num_qkv_heads * head_dim).astype(np.float32)
x_jax = jnp.array(x_np)
x_torch = torch.tensor(x_np)

# Test split_heads
(jax_q, jax_k, jax_v) = split_heads(x_jax, config)
(torch_q, torch_k, torch_v) = torch_attn._split_heads(x_torch)
noise = np.random.randn(jax_q.shape[0], jax_q.shape[1], jax_q.shape[2], jax_q.shape[3]).astype(np.float32)
compare_results(jax_q, jnp.array(torch_q.numpy(), dtype=jnp.float32))
compare_results(jax_k, jnp.array(torch_k.numpy(), dtype=jnp.float32))
compare_results(jax_v, jnp.array(torch_v.numpy(), dtype=jnp.float32))


x_np = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
x_jax = jnp.array(x_np)
x_torch = torch.tensor(x_np)

attn_jax = AttentionLayer(config=config)
attn_torch = FalconAttention(config)
attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
attention_mask_torch = torch.ones((batch_size, seq_len), dtype=torch.float32)
position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
position_ids_torch = torch.Tensor(np.array(position_ids)).to(torch.int32)
variables = attn_jax.init(jax.random.PRNGKey(0), x_jax, attention_mask, position_ids)
params = variables['params']
print(f"jax: {params['query_key_value']['kernel'].shape} torch: {attn_torch.query_key_value.weight.shape}")
print(f"jax: {params['dense']['kernel'].shape} torch: {attn_torch.dense.weight.shape}")
params['query_key_value']['kernel'] = jnp.array(attn_torch.query_key_value.weight.detach().numpy())
#params['query_key_value']['bias'] = jnp.array(attn_torch.query_key_value.bias.detach().numpy()) if attn_torch.query_key_value.bias is not None else None
params['dense']['kernel'] = jnp.array(attn_torch.dense.weight.detach().numpy())
#params['dense']['bias'] = jnp.array(attn_torch.dense.bias.detach().numpy()) if attn_torch.dense.bias is not None else None

torch_rope = FalconRotaryEmbedding(config)
pos_embeddings = torch_rope(x_torch, position_ids_torch)
attention_mask_torch = attention_mask_torch[:, None, None, :]

# Forward
out_jax, _, jax_scores = attn_jax.apply({'params': params}, x_jax, attention_mask, position_ids, output_attentions=True)
out_torch, _, torch_scores = attn_torch(x_torch, alibi=None, attention_mask=attention_mask_torch, position_embeddings=pos_embeddings, output_attentions=True)
compare_results(out_jax, jnp.array(out_torch.detach().numpy(), dtype=jnp.float32))
compare_results(jax_scores, jnp.array(torch_scores.detach().numpy(), dtype=jnp.float32))

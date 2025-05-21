import jax
import numpy as np
from transformers.models.falcon.modeling_falcon import (
    FalconMLP,
    FalconAttention,
)
from model.model_falcon import merge_heads, split_heads, MLPBlock 
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

mlp_jax = MLPBlock(config=config)
mlp_torch = FalconMLP(config)
vars = mlp_jax.init(jax.random.PRNGKey(0), x_jax)

vars['params']['dense_h_to_4h']['kernel'] = jnp.array(mlp_torch.dense_h_to_4h.weight.detach().numpy())
vars['params']['dense_4h_to_h']['kernel'] = jnp.array(mlp_torch.dense_4h_to_h.weight.detach().numpy())

out_jax = mlp_jax.apply(vars, x_jax)
out_torch = mlp_torch(x_torch)
compare_results(out_jax, jnp.array(out_torch.detach().numpy()))
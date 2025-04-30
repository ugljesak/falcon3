import jax
import numpy as np
from transformers.models.falcon.modeling_falcon import (
    FalconMLP,
    FalconAttention,
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
hidden_size = config.hidden_size

x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
x_jax = jnp.array(x_np)
x_torch = torch.tensor(x_np)


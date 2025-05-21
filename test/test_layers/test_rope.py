import jax
import numpy as np
import sys
sys.path.append("transformers/src/transformers/models/falcon")
from transformers.models.falcon.modeling_falcon import (
    FalconLinear,
    rotate_half as torch_rotate_half,
    apply_rotary_pos_emb as torch_apply_rotary_pos_emb,
    FalconRotaryEmbedding
)
from transformers.models.falcon.configuration_falcon import (
    FalconConfig, 
)
from model.model_falcon import RotaryPositionEmbedding, apply_rotary_pos_emb, rotate_by_quarter
import jax.numpy as jnp
import torch
from .test_utils import compare_results
from model.configuration_falcon import FalconConfig

batch_size = 8
input_dim = 128
head_dim = 64
output_dim = 128
# Function to create a simple JAX dense layer
rope_config = FalconConfig(group_query=True)
x_torch = torch.ones(batch_size, input_dim, dtype=torch.float32)
x_jax = jnp.array(x_torch.numpy(), dtype=jnp.float32)

rotate_half_jax = rotate_by_quarter(x_jax)
rotate_half_torch = torch_rotate_half(x_torch)
compare_results(rotate_half_jax, jnp.array(rotate_half_torch.numpy(), dtype=jnp.float32))

position_ids = jnp.arange(input_dim)[None, :].repeat(batch_size, axis=0)
position_ids_torch = torch.Tensor(position_ids).to(torch.int32)
jax_rope = RotaryPositionEmbedding(rope_config)
variables = jax_rope.init(jax.random.PRNGKey(0), x_jax, position_ids)
torch_rope = FalconRotaryEmbedding(rope_config)

cos_jax, sin_jax = jax_rope.apply(variables, x_jax, position_ids)
cos_torch, sin_torch = torch_rope(x_torch, position_ids_torch)
compare_results(cos_jax, jnp.array(cos_torch.numpy(), dtype=jnp.float32))
compare_results(sin_jax, jnp.array(sin_torch.numpy(), dtype=jnp.float32))

q_jax = jnp.ones((batch_size, input_dim, head_dim), dtype=jnp.float32)
k_jax = jnp.ones((batch_size, input_dim, head_dim), dtype=jnp.float32)

q_torch = torch.ones((batch_size, input_dim, head_dim), dtype=torch.float32)
k_torch = torch.ones((batch_size, input_dim, head_dim), dtype=torch.float32)

q_embed_torch, k_embed_torch = torch_apply_rotary_pos_emb(q_torch, k_torch, cos_torch, sin_torch)
q_embed_jax, k_embed_jax = apply_rotary_pos_emb(q_jax, k_jax, cos_jax, sin_jax)
print(q_embed_torch)
compare_results(q_embed_jax, jnp.array(q_embed_torch.numpy(), dtype=jnp.float32))
compare_results(k_embed_jax, jnp.array(k_embed_torch.numpy(), dtype=jnp.float32))

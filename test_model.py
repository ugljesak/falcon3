import jax
import numpy as np
from transformers.models.falcon.modeling_falcon import (
    FalconRotaryEmbedding,
    FalconDecoderLayer,
    FalconModel as FalconTorch
)
from model import MLPBlock, AttentionLayer, DecoderLayer, FalconModel
import jax.numpy as jnp
import torch
from utils import compare_results
from configuration_falcon import FalconConfig

config = FalconConfig(
    group_query=True,
    num_kv_heads=1,
    num_attention_heads=30,
    hidden_size=480,
    num_hidden_layers=4,
    num_ln_in_parallel_attn=2,
    parallel_attn=True,
)

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

falcon_jax = FalconModel(config=config)
falcon_torch = FalconTorch(config)

# Initialize the JAX model
# Initialize JAX model parameters
key = jax.random.PRNGKey(0)
params_jax = falcon_jax.init(key, x_jax, attention_mask_jax, position_ids_jax)

# Copy weights from torch to jax
def torch_to_numpy(tensor):
    return np.asarray(tensor.detach().cpu().numpy())

def copy_params(torch_model, jax_params):
    new_params = {}
    for k, v in jax_params.items():
        if isinstance(v, dict):
            new_params[k] = copy_params(getattr(torch_model, k, torch_model.state_dict().get(k, {})), v)
        else:
            torch_val = torch_model.state_dict().get(k, None)
            if torch_val is not None:
                new_params[k] = jnp.array(torch_to_numpy(torch_val))
            else:
                new_params[k] = v
    return new_params

params_jax = copy_params(falcon_torch, params_jax)

# Apply the parameters to the JAX model
out_jax, kv_cache, attn_out_jax = falcon_jax.apply(
    params_jax,
    x_jax,
    attention_mask=attention_mask_jax,
    position_ids=position_ids_jax,
    output_attentions=True,
    use_cache=True
)
out_torch, cache, attn_out_torch = falcon_torch(
    x_torch,
    attention_mask=attention_mask_torch[:, None, None, :],
    position_ids=position_ids_torch,
    position_embeddings=None,
    output_attentions=True,
    use_cache=True
)
compare_results(out_jax, jnp.array(out_torch.detach().numpy()))
compare_results(attn_out_jax, jnp.array(attn_out_torch.detach().numpy()))
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
from output_models import *

config = FalconConfig(
    group_query=True,
    num_hidden_layers=4,
    num_ln_in_parallel_attn=2,
    parallel_attn=True,
)

batch_size = 8
seq_len = 50
num_heads = config.num_attention_heads
head_dim = config.head_dim
hidden_size = config.hidden_size

x_np = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.int32)
x_jax = jnp.array(x_np)
x_torch = torch.tensor(x_np)
attention_mask_jax = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
attention_mask_torch = torch.ones((batch_size, seq_len), dtype=torch.float32)
position_ids_jax = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
position_ids_torch = torch.Tensor(np.array(position_ids_jax)).to(torch.int32)

falcon_jax = FalconModel(config=config)
falcon_torch = FalconTorch(config)

# Copy weights from torch to jax
def torch_to_jnp(tensor):
    return jnp.array(tensor.detach().numpy())

# Initialize the JAX model
# Initialize JAX model parameters
key = jax.random.PRNGKey(0)
def init_jax_model(key, x_jax, attention_mask_jax, position_ids_jax, falcon_torch):
    params_jax = falcon_jax.init(key, input_ids=x_jax, input_embeds=None, attention_mask=attention_mask_jax, position_ids=position_ids_jax)
    params_jax['params']['word_embeddings']['embedding'] = torch_to_jnp(falcon_torch.word_embeddings.weight)
    for i in range(config.num_hidden_layers):
        blocks_i = "blocks_" + str(i)
        params_jax['params'][blocks_i]['attention']['query_key_value']['kernel'] = torch_to_jnp(falcon_torch.h[i].self_attention.query_key_value.weight)
        params_jax['params'][blocks_i]['attention']['dense']['kernel'] = torch_to_jnp(falcon_torch.h[i].self_attention.dense.weight)
        params_jax['params'][blocks_i]['mlp']['dense_h_to_4h']['kernel'] = torch_to_jnp(falcon_torch.h[i].mlp.dense_h_to_4h.weight)
        params_jax['params'][blocks_i]['mlp']['dense_4h_to_h']['kernel'] = torch_to_jnp(falcon_torch.h[i].mlp.dense_4h_to_h.weight)
    return params_jax

params_jax = init_jax_model(key, x_jax, attention_mask_jax, position_ids_jax, falcon_torch)
print(f"jax: {params_jax['params']['word_embeddings']['embedding'].shape} torch: {falcon_torch.word_embeddings.weight.shape}")

# jax_embeddings_params = {'params': {'word_embeddings': {'embedding': params_jax['params']['word_embeddings']['embedding']}}}
# @jax.jit
# def apply_jax_embeddings(x_jax, params_jax):
#     return falcon_jax.apply(params_jax, x_jax, method=lambda mdl, ids: mdl.word_embeddings(ids))

# jax_embeddings = apply_jax_embeddings(x_jax, jax_embeddings_params)    
# torch_embeddings = falcon_torch.word_embeddings(x_torch)
# compare_results(jax_embeddings, jnp.array(torch_embeddings.detach().numpy(), dtype=jnp.float32))

# Apply the parameters to the JAX model

@jax.jit
def jax_apply(x_jax, params_jax, attention_mask_jax, position_ids_jax):
    out = falcon_jax.apply(
        params_jax,
        x_jax,
        attention_mask=attention_mask_jax,
        position_ids=position_ids_jax,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=True
    )
    return out.last_hidden_state, out.past_key_values, out.attentions

out_jax = jax_apply(x_jax, params_jax, attention_mask_jax, position_ids_jax)
out_torch = falcon_torch(
    x_torch,
    attention_mask=attention_mask_torch,
    position_ids=position_ids_torch,
    output_attentions=True,
    output_hidden_states=True,
    use_cache=True
)
# return BaseModelOutputWithPastAndCrossAttentions(
#                 last_hidden_state=hidden_states,
#                 past_key_values=next_cache,
#                 hidden_states=all_hidden_states,
#                 attentions=all_self_attentions
#             )
print(f"jax: {len(out_jax[2])} torch: {len(out_torch.attentions)}")
compare_results(out_jax[0], jnp.array(out_torch.last_hidden_state.detach().numpy()))

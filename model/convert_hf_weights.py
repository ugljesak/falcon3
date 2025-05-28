import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional
from .model_falcon import FalconForCausalLM
from .configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import FalconForCausalLM as TorchFalconForCausalLM

def torch_to_jnp(tensor):
    # Convert the PyTorch tensor to a NumPy array
    np_array = tensor.detach().cpu().numpy()
    
    # Convert the NumPy array to a JAX array
    jax_array = jnp.array(np_array)
    
    return jax_array

def make_model(
    config: FalconConfig,
    torch_model: TorchFalconForCausalLM,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    input_ids: Optional[jax.Array] = None,
    attention_mask: Optional[jax.Array] = None
    ) -> FalconForCausalLM:
    """
    Convert a Hugging Face Falcon model to a JAX Falcon model.
    """
    jax_model = FalconForCausalLM(config=config)    
    
    # Initialize dummy JAX model parameters
    # This is necessary to create the model structure
    # and to ensure that the parameters are in the correct format
    # batch_size = 2
    # seq_len = 50
    # x_jax = jnp.array(np.random.randint(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.int32))
    # attention_mask_jax = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    position_ids_jax = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    params_jax = jax_model.init(jax.random.PRNGKey(69), input_ids=input_ids, input_embeds=None, attention_mask=attention_mask, position_ids=position_ids_jax)

    # Copy weights from torch to jax
    params_jax['params']['transformer']['word_embeddings']['embedding'] = torch_to_jnp(torch_model.transformer.word_embeddings.weight)
    for i in range(config.num_hidden_layers):
        blocks_i = "blocks_" + str(i)
        params_jax['params']['transformer'][blocks_i]['attention']['query_key_value']['kernel'] = torch_to_jnp(torch_model.transformer.h[i].self_attention.query_key_value.weight)
        params_jax['params']['transformer'][blocks_i]['attention']['dense']['kernel'] = torch_to_jnp(torch_model.transformer.h[i].self_attention.dense.weight)
        params_jax['params']['transformer'][blocks_i]['mlp']['dense_h_to_4h']['kernel'] = torch_to_jnp(torch_model.transformer.h[i].mlp.dense_h_to_4h.weight)
        params_jax['params']['transformer'][blocks_i]['mlp']['dense_4h_to_h']['kernel'] = torch_to_jnp(torch_model.transformer.h[i].mlp.dense_4h_to_h.weight)
    params_jax['params']['lm_head']['kernel'] = torch_to_jnp(torch_model.lm_head.weight)

    return jax_model, params_jax
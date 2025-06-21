import numpy as np
import jax
import jax.numpy as jnp
import torch
from typing import Optional
from .model_falcon3 import FlaxFalcon3ForCausalLM
from .configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import FalconForCausalLM as TorchFalconForCausalLM

def torch_to_jnp(tensor):
    # Special handling for bfloat16 tensors
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float().detach().cpu().numpy()
        tensor = jnp.array(tensor)
    else:
        tensor = tensor.detach().cpu().numpy()
        tensor = jnp.array(tensor)

    return tensor

def make_model(config, torch_model, batch_size, seq_len, input_ids, attention_mask):
    """
    Convert a Hugging Face Falcon model to a JAX Falcon model.
    """
    print("🖨️  Converting weights...")
    flax_model = FlaxFalcon3ForCausalLM(config=config)    
    
    # Initialize dummy JAX model parameters
    # This is necessary to create the model structure
    # and to ensure that the parameters are in the correct format
    # batch_size = 2
    # seq_len = 50
    # x_jax = jnp.array(np.random.randint(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.int32))
    # attention_mask_jax = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    position_ids_jax = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    flax_params = flax_model.init(jax.random.PRNGKey(69), input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids_jax, init_cache=True, return_dict=True)

    #print(params_jax['params']['transformer']['blocks_0'].keys())
    #print(dir(torch_model.transformer.h[0]))
    # Copy weights from torch to jax
    flax_params['params']['model']['embed_tokens']['embedding'] = torch_to_jnp(
        torch_model.model.embed_tokens.weight
    )
    for i in range(config.num_hidden_layers):
        flax_params['params']['model']['layers'][f'{i}']['input_layernorm']['weight'] = torch_to_jnp(
            torch_model.model.layers[i].input_layernorm.weight
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['q_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.q_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['k_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.k_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['v_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.v_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['o_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.o_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['post_attention_layernorm']['weight'] = torch_to_jnp(
            torch_model.model.layers[i].post_attention_layernorm.weight
        )
        flax_params['params']['model']['layers'][f'{i}']['mlp']['up_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].mlp.up_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['mlp']['gate_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].mlp.gate_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['mlp']['down_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].mlp.down_proj.weight.T
        )
    flax_params['params']['model']['norm']['weight'] = torch_to_jnp(
        torch_model.model.norm.weight
    )
    flax_params['params']['lm_head']['kernel'] = torch_to_jnp(
        torch_model.lm_head.weight.T
    )

    return flax_model, flax_params, flax_params['cache']
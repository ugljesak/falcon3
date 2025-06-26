import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import torch
import json
from pathlib import Path
from typing import Optional
from .model_falcon3 import FlaxFalcon3ForCausalLM
from .configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import FalconForCausalLM as TorchFalconForCausalLM
from safetensors.torch import load_file as safe_load_file

def torch_to_jnp(tensor):
    # Special handling for bfloat16 tensors
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float().detach().cpu().numpy()
            tensor = jnp.array(tensor)
        else:
            tensor = tensor.detach().cpu().numpy()
            tensor = jnp.array(tensor)
    else:
        if isinstance(tensor, np.ndarray):
            tensor = jnp.array(tensor)

    return tensor

def convert_from_hf_weights(checkpoint_path, batch_size, seq_len, config):
    ckpt_paths = sorted(Path(checkpoint_path).glob("*.safetensors"))
    ckpts = []
    for i, ckpt_path in enumerate(ckpt_paths):
        checkpoint = safe_load_file(str(ckpt_path))
        # for key, value in checkpoint.items():
        #     if isinstance(value, torch.Tensor):
        #         checkpoint[key] = torch_to_jnp(value)
        #     elif isinstance(value, np.ndarray):
        #         checkpoint[key] = torch_to_jnp(value)
        #     elif isinstance(value, dict):
        #         for sub_key, sub_value in value.items():
        #             if isinstance(sub_value, torch.Tensor):
        #                 value[sub_key] = torch_to_jnp(sub_value)
        #             elif isinstance(sub_value, np.ndarray):
        #                 value[sub_key] = torch_to_jnp(sub_value)
        #breakpoint()  # Debugging point to inspect checkpoint
        ckpts.append(checkpoint)

    def from_checkpoint(key, axis=0):
        weights = jnp.concatenate([torch_to_jnp(ckpt[key]) for ckpt in ckpts if key in ckpt], axis=axis) 
        return weights

    print(f"Loaded {len(ckpts)} checkpoints from {checkpoint_path}")
    flax_params = {
        'params': {
            'model': {
                'embed_tokens': {'embedding': torch_to_jnp(ckpts[0]['model.embed_tokens.weight'])}, 
                'norm': {'weight': torch_to_jnp(ckpts[2]['model.norm.weight'])}, 
                'layers': {
                    f'{layer}': {
                        'input_layernorm': {'weight': torch_to_jnp(ckpts[0][f'model.layers.{layer}.input_layernorm.weight'])},
                        'self_attn': {
                            'q_proj': {'kernel': from_checkpoint(f'model.layers.{layer}.self_attn.q_proj.weight', axis=0).transpose()}, 
                            'k_proj': {'kernel': from_checkpoint(f'model.layers.{layer}.self_attn.k_proj.weight', axis=0).transpose()},
                            'v_proj': {'kernel': from_checkpoint(f'model.layers.{layer}.self_attn.v_proj.weight', axis=0).transpose()},
                            'o_proj': {'kernel': from_checkpoint(f'model.layers.{layer}.self_attn.o_proj.weight', axis=1).transpose()}, 
                        }, 
                        'post_attention_layernorm': {'weight': torch_to_jnp(ckpts[0][f'model.layers.{layer}.post_attention_layernorm.weight'])},
                        'mlp': {
                            'up_proj': {'kernel': from_checkpoint(f'model.layers.{layer}.mlp.up_proj.weight', axis=0).transpose()}, 
                            'gate_proj': {'kernel': from_checkpoint(f'model.layers.{layer}.mlp.gate_proj.weight', axis=0).transpose()},
                            'down_proj': {'kernel': from_checkpoint(f'model.layers.{layer}.mlp.down_proj.weight', axis=1).transpose()},
                        }, 
                    }
                for layer in range(config.num_hidden_layers)}, 
            }, 
            'lm_head': {'kernel': torch_to_jnp(ckpts[3]['lm_head.weight']).transpose()}, 
        },
        'cache': {
            'model': {
                'layers': {
                    f'{layer}': {
                        'self_attn': {
                            'cached_key': jnp.zeros((batch_size, seq_len, config.num_key_value_heads, config.head_dim), dtype=jnp.float32),
                            'cached_value': jnp.zeros((batch_size, seq_len, config.num_key_value_heads, config.head_dim), dtype=jnp.float32),
                            'cache_index': jnp.array(0, dtype=jnp.int32), 
                        }
                    }
                for layer in range(config.num_hidden_layers)}, 
            }
        }
    }
    del ckpts
    return flax_params

def convert_from_torch_model(torch_model, flax_model, batch_size, seq_len, config):

    input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    position_ids_jax = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)


    flax_init_jit = jax.jit(flax_model.init, static_argnames=('init_cache', 'return_dict'))
    flax_params = flax_init_jit(jax.random.PRNGKey(69), input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids_jax, init_cache=True, return_dict=True)
    breakpoint()  # Debugging point to inspect flax_params
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
    return flax_model, flax_params

def make_model(config, torch_model, batch_size, seq_len, rule):
    """
    Convert a Hugging Face Falcon model to a JAX Falcon model.
    """
    flax_model = FlaxFalcon3ForCausalLM(config=config)
    print("🖨️  Converting weights...")
    
    if(rule == 'torch'):
        flax_model, flax_params = convert_from_torch_model(torch_model, flax_model, batch_size, seq_len, config)
    elif(rule == 'hf'):
        path = '../.cache/huggingface/hub/models--tiiuae--Falcon3-7B-Instruct/snapshots/1e57a0ecd176c7c139f289c60a74e57f887c3dfb/'
        flax_params = convert_from_hf_weights(path, batch_size, seq_len, config)
    else:
        raise ValueError(f"Unknown conversion rule: {rule}. Use 'torch' or 'hf'.")

    return flax_model, flax_params
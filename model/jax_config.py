import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
from model.model_falcon3 import FlaxFalcon3ForCausalLM
from model.configuration_falcon3 import Falcon3Config

global DEVICE_COUNT
DEVICE_COUNT = None  # Set the number of devices for parallelism
def set_device_count(count):
    """Set the global device count for JAX."""
    global DEVICE_COUNT
    DEVICE_COUNT = count
    os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={DEVICE_COUNT}'
    jax.config.update("jax_platform_name", "cpu")  # Ensure JAX uses CPU for testing

def create_device_mesh(dp_size, tp_size):
    """Create a 2D device mesh for tensor and data parallelism."""
    if DEVICE_COUNT is None:
        set_device_count(dp_size * tp_size)
    if dp_size * tp_size != DEVICE_COUNT:
        print(f"Warning: Device count mismatch! Expected {DEVICE_COUNT}, got {dp_size * tp_size}.")
        print("Setting device count to dp_size * tp_size.")
        set_device_count(dp_size * tp_size)
    devices = np.array(jax.devices()).reshape(dp_size, tp_size)
    return Mesh(devices, ('dp', 'tp'))

def with_named_sharding_constraint(x, mesh, partition_spec):
    if mesh is not None:
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, partition_spec))
    else:
        print("No mesh defined, skipping sharding constraint.")
        return x
    
def get_partitioning_rules(config: Falcon3Config = None):
    
    partitioning_rules = {
        'params': {
            'model': {
                'embed_tokens': {'embedding': P('tp', None)}, 
                'norm': {'weight': P()}, 
                'layers': {
                    f'{layer}': {
                        'input_layernorm': {'weight': P()},
                        'self_attn': {
                            'q_proj': {'kernel': P(None, 'tp')}, 
                            'k_proj': {'kernel': P(None, 'tp')}, 
                            'v_proj': {'kernel': P(None, 'tp')}, 
                            'o_proj': {'kernel': P('tp', None)}, 
                        }, 
                        'post_attention_layernorm': {'weight': P()},
                        'mlp': {
                            'up_proj': {'kernel': P(None, 'tp')}, 
                            'gate_proj': {'kernel': P(None, 'tp')}, 
                            'down_proj': {'kernel': P('tp', None)}, 
                        }, 
                    }
                for layer in range(config.num_hidden_layers)}, 
            }, 
            'lm_head': {'kernel': P(None, 'tp')},
        },
        'cache': {
            'model': {
                'layers': {
                    f'{layer}': {
                        'self_attn': {
                            'cached_key': P(),
                            'cached_value': P(),
                            'cache_index': P(), 
                        }
                    }
                for layer in range(config.num_hidden_layers)}, 
            }
        }
    }
    return partitioning_rules

def shard_params(params, rules, device_mesh):
    """Apply sharding to loaded parameters based on partitioning rules."""
    params = flatten_dict(params)
    rules = flatten_dict(rules)
    
    sharded_params = {}

    for param_key, param_value in params.items():
        # Find the corresponding rule
        rule_key = param_key  # Adjust if your rules have different structure
        if rule_key in rules:
            partition_spec = rules[rule_key]
            
            sharding = NamedSharding(device_mesh, partition_spec)
            sharded_param = jax.device_put(param_value, sharding)
            sharded_params[param_key] = sharded_param
        else:
            sharded_params[param_key] = param_value
    
    return unflatten_dict(sharded_params)

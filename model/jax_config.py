import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from model.model_falcon3 import FlaxFalcon3ForCausalLM
from model.configuration_falcon3 import Falcon3Config
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
def create_device_mesh(dp_size, tp_size):
    """Create a 2D device mesh for tensor and data parallelism."""
    devices = np.array(jax.devices()).reshape(dp_size, tp_size)
    return Mesh(devices, ('dp', 'tp'))




def init_sharded_model(config, rng, mesh):
    """Initialize the model with proper sharding."""
    tp_size = mesh.shape['tp']
    dp_size = mesh.shape['dp']
    
    input_shape = (8, 128)  # Batch size, sequence length
    input_ids = jnp.zeros(input_shape, dtype=jnp.int32)
    attention_mask = jnp.ones(input_shape, dtype=jnp.int32)
    position_ids = jnp.broadcast_to(
        jnp.arange(input_shape[1])[None, :], 
        input_shape
    )
    
    # Create model
    model = FlaxFalcon3ForCausalLM(
        config=config, 
        dtype=jnp.float32,
        tp_size=tp_size,
        dp_size=dp_size
    )
    
    # Initialize parameters with proper sharding
    with mesh:
        params = jax.jit(
            model.init, 
            in_shardings=(None, None, None, None),  # RNG, input_ids, attention_mask, position_ids
            out_shardings=None  # Output params (will be annotated later)
        )(rng, input_ids, attention_mask, position_ids)
        
        # Get sharding rules
        #rules = get_sharding_annotations(model)
        
        # Apply sharding rules to parameters
        flat_params = flatten_dict(params)
        sharded_flat_params = {}
        
        for key, param in flat_params.items():
            param_name = '/'.join([str(k) for k in key])
            if param_name in rules:
                spec = rules[param_name]
            else:
                # Default: replicate
                spec = P()
            
            # Shard the parameter
            sharded_flat_params[key] = jax.device_put(param, jax.sharding.NamedSharding(mesh, spec))
        
        # Reconstruct parameter tree
        sharded_params = unflatten_dict(sharded_flat_params)
    
    return model, sharded_params
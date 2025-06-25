import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
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


def create_device_mesh(dp_size, tp_size):
    """Create a 2D device mesh for tensor and data parallelism."""
    devices = np.array(jax.devices()).reshape(dp_size, tp_size)
    return Mesh(devices, ('dp', 'tp'))

def with_named_sharding_constraint(x, mesh, partition_spec):
    if mesh is not None:
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, partition_spec))
    else:
        print("No mesh defined, skipping sharding constraint.")
        return x
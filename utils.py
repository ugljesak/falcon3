import jax
import inspect
import jax.numpy as jnp
from typing import Any, NamedTuple, Optional, Tuple

def compare_results(x, y):
    frame = inspect.currentframe().f_back
    # Find the line in the source code where compare_results is called
    lines, lineno = inspect.getsourcelines(frame)
    line = lines[frame.f_lineno - lineno - 1].strip()
    pos = line.find('(')
    line = line[pos+1:-1]
    pos = line.find(',')
    x_name = line[:pos].strip() if pos is not None else 'x'
    y_name = line[pos+1:].strip() if pos is not None else 'y'
    
    print(
f"""Info for first argument: 
    Name: {x_name},
    Class: {x.__class__},
    Shape: {x.shape}.""")
    print(
f"""Info for second argument:
    Name: {y_name},
    Class: {y.__class__},
    Shape: {y.shape}.""")
    print(f"PCC Score: {jnp.min(jnp.corrcoef(x.flatten(), y.flatten()))}")
    print(f"Max Difference: {jnp.max(jnp.abs(x - y))}")
    print("=" * 50)

class KVCache():
    k_cache: jax.Array
    v_cache: jax.Array

    def __init__(self, config, key: Optional[jax.Array] = None, value: Optional[jax.Array] = None):
        """Initialize the KVCache with key and value tensors."""
        # key and value shapes: [batch_size, num_heads, max_seq_len, head_dim]
        if key is None:
            key = jnp.zeros((config.batch_size, config.num_attention_heads, config.max_position_embeddings, config.head_dim), dtype=config.dtype)
        else:
            assert key.ndim == 4, f"Key should be 4D, but got {key.ndim}D."
            assert value.ndim == 4, f"Value should be 4D, but got {value.ndim}D."
            if key.shape != (config.batch_size, config.num_attention_heads, config.max_position_embeddings, config.head_dim):
                # Expand key to the correct shape, keeping existing values and padding with zeros if needed
                pad_shape = (
                    0, config.batch_size - key.shape[0],
                    0, config.num_attention_heads - key.shape[1],
                    0, config.max_position_embeddings - key.shape[2],
                    0, config.head_dim - key.shape[3],
                )
                key = jnp.pad(
                    key,
                    ((0, pad_shape[1]), (0, pad_shape[3]), (0, pad_shape[5]), (0, pad_shape[7])),
                    mode="constant"
                )
        self.key = key
        if value is None:
            value = jnp.zeros((config.batch_size, config.num_attention_heads, config.max_position_embeddings, config.head_dim), dtype=config.dtype)
        else:
            assert value.ndim == 4, f"Value should be 4D, but got {value.ndim}D."
            if value.shape != (config.batch_size, config.num_attention_heads, config.max_position_embeddings, config.head_dim):
                # Expand value to the correct shape, keeping existing values and padding with zeros if needed
                pad_shape = (
                    0, config.batch_size - value.shape[0],
                    0, config.num_attention_heads - value.shape[1],
                    0, config.max_position_embeddings - value.shape[2],
                    0, config.head_dim - value.shape[3],
                )
                value = jnp.pad(
                    value,
                    ((0, pad_shape[1]), (0, pad_shape[3]), (0, pad_shape[5]), (0, pad_shape[7])),
                    mode="constant"
                )
        self.value = value

    def shift_left_kv_cache(self):
        """Shift the key and value cache to the left by one position."""
        # Shift by -1 (1 to the left) along the second last axis (`seq_len`)
        k_cache = jnp.roll(k_cache, shift=-1, axis=-2)
        v_cache = jnp.roll(v_cache, shift=-1, axis=-2) 

    def update(self, new_key, new_value, cache_position, sin : Optional[jax.Array] = None, cos: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array]:
        """Update the cache with new key and value."""
        key = self.key.at[:, :, cache_position, :].set(new_key)
        value = self.value.at[:, :, cache_position, :].set(new_value)
        return key, value
import jax
import inspect
import jax.numpy as jnp
import optax
from typing import Any, NamedTuple, Optional, Tuple
from configuration_falcon import FalconConfig

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
    print(f"Mean Difference: {jnp.mean(jnp.abs(x - y))}")
    print("=" * 50)

class KVCache():
    k_cache: jax.Array
    v_cache: jax.Array

    def __init__(self, config: FalconConfig, batch_size: int, key: Optional[jax.Array] = None, value: Optional[jax.Array] = None):
        """Initialize the KVCache with key and value tensors."""
        # key and value shapes: [batch_size, num_heads, max_seq_len, head_dim]
        if key is None:
            self.key = jnp.zeros((batch_size, config.num_attention_heads, config.max_position_embedding, config.head_dim), dtype=jnp.float32)
        if value is None:
            self.value = jnp.zeros((batch_size, config.num_attention_heads, config.max_position_embedding, config.head_dim), dtype=jnp.float32)
        self.k_cache = key
        self.v_cache = value
        

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
    

def fixed_cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[jax.Array] = None,
) -> jax.Array:
    """Compute the cross-entropy loss with fixed logits."""

    if shift_labels is None:
        # Shift so that tokens < n predict n
        pad = jnp.full(labels.shape[:-1] + (1,), ignore_index, dtype=labels.dtype)
        labels = jnp.concatenate([labels, pad], axis=-1)
        shift_labels = labels[..., 1:]

    # Flatten the tokens
    logits = logits.reshape(-1, vocab_size)
    shift_labels = shift_labels.reshape(-1)

    mask = (shift_labels != ignore_index)
    masked_labels = jnp.where(mask, shift_labels, 0)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, masked_labels, axis=-1)
    loss *= mask # Apply mask to loss
    
    if num_items_in_batch is not None:
        # If num_items_in_batch is provided, use it for normalization
        loss = loss.sum() / num_items_in_batch
    else:
        # Otherwise, use the sum of the mask for normalization
        loss = loss.sum() / jnp.maximum(mask.sum(), 1)

    return loss
import jax
import jax.numpy as jnp
from .model_falcon import FalconForCausalLM

def generate(
    model: FalconForCausalLM,
    params,
    input_ids,
    attention_mask=None,
    max_new_tokens=20,
    pad_token_id=None,
    eos_token_id=None,
    position_ids = None,
):
    """Generate text efficiently using properly initialized KV caching in NN."""
    # Set defaults for special tokens
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    
    # Initialize generation variables
    batch_size, seq_length = input_ids.shape
    print(f"Generatiing with input_ids shape {input_ids.shape}")
    max_length = seq_length + max_new_tokens
    has_reached_eos = jnp.zeros(batch_size, dtype=jnp.bool_)
    
    
    position_ids = jnp.cumsum(attention_mask, axis=-1) - 1 if position_ids is None else position_ids
    # Now process the real prompt to fill in the cache for actual tokens
    print("First pass to fill cache with input tokens")
    print(f"Input IDs shape: {input_ids.shape}, Position IDs shape: {position_ids.shape}")
    print(f"Attention Mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")
    print(f"Position IDs: {position_ids}")
    outputs = model.apply(
        params,
        input_ids=input_ids,
        position_ids=position_ids[:, :seq_length],  # Use only the initial sequence length
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    # Get next token prediction
    next_token_logits = outputs.logits[:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    next_token = next_token[:, None]  # Add sequence dimension
    
    # # Add first generated token
    all_token_ids = jnp.concatenate([input_ids, next_token], axis=1)
    
    # Track current sequence length
    
    # Check early stopping conditions
    if eos_token_id is not None:
        has_reached_eos = has_reached_eos | (next_token[:, 0] == eos_token_id)

    cur_len = seq_length
    # Start auto-regressive generation loop
    for i in range(1, max_new_tokens):
        # Early exit if all sequences have reached EOS
        print("------", i, "------") #just to track tokens
        if eos_token_id is not None and jnp.all(has_reached_eos):
            break
        
        past_key_values = outputs.past_key_values
        # Generate next token using cache
        new_position_ids = position_ids[:, cur_len].reshape(-1, 1)
        print(f"New position IDs: {new_position_ids}")
        outputs = model.apply(
            params,
            input_ids=next_token,  # Only process the new token
            attention_mask=attention_mask,
            use_cache=True,
            kv_cache=past_key_values,
            position_ids = new_position_ids,
        )
        cur_len += 1
        # Get logits and predict next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token = next_token[:, None]  # Add sequence dimension
        # Add new token to results
        all_token_ids = jnp.concatenate([all_token_ids, next_token], axis=1)
        
        # Update EOS tracking
        if eos_token_id is not None:
            has_reached_eos = has_reached_eos | (next_token[:, 0] == eos_token_id)
    
    return all_token_ids
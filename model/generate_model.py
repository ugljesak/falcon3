import jax
import jax.numpy as jnp
import numpy as np
import torch
from .model_falcon import FalconForCausalLM
from test.test_utils import compare_results
from model.convert_hf_weights import torch_to_jnp


@jax.jit
def generate(
    model: FalconForCausalLM,
    params,
    input_ids,
    attention_mask=None,
    max_new_tokens=20,
    pad_token_id=None,
    eos_token_id=None,
    past_key_values=None,
    position_ids = None,
):
    """Generate text efficiently using properly initialized KV caching in NN."""
    # Set defaults for special tokens
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    
    # Initialize generation variables
    batch_size, seq_length = input_ids.shape
    has_reached_eos = jnp.zeros(batch_size, dtype=jnp.bool_)
    
    position_ids = jnp.cumsum(attention_mask, axis=-1) - 1 if position_ids is None else position_ids
    print(f"Initial pass for loading cache into model...")
    outputs, cache = model.apply(
        {'params': params['params'], 'cache': past_key_values},
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids[:, :seq_length],
        return_dict=True,
        #init_cache=True,
        mutable=['cache'],
    )
    #breakpoint()  # For debugging purposes, you can remove this line later
    
    # Get cache, logits and predict next token
    next_token_logits = outputs['logits'][:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    next_token = next_token[:, None]  # Add sequence dimension
    all_token_ids = jnp.concatenate([input_ids, next_token], axis=1)
    cache = cache['cache']
    if eos_token_id is not None:
        has_reached_eos |= (next_token[:, 0] == eos_token_id)

    
    print(f"Auto-regressive generation to predict next tokens...")
    cur_len = seq_length
    # Start auto-regressive generation loop

    for i in range(1, max_new_tokens):
        print("------", i, "------") #just to track tokens
        if eos_token_id is not None and jnp.all(has_reached_eos):
            break
        
        outputs, cache = model.apply(
            {'params': params['params'], 'cache': cache},
            input_ids=next_token,  # Only process the new token
            attention_mask=attention_mask,
            position_ids = position_ids[:, cur_len].reshape(-1, 1),
            return_dict=True,
            mutable=['cache'],
        )
        # Get cache, logits and predict next token
        next_token_logits = outputs['logits'][:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token = next_token[:, None]  # Add sequence dimension
        all_token_ids = jnp.concatenate([all_token_ids, next_token], axis=1)
        cache = cache['cache']
        if eos_token_id is not None:
            has_reached_eos |= (next_token[:, 0] == eos_token_id)
        
        cur_len += 1
    
    return all_token_ids
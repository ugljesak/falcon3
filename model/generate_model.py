import jax
import jax.numpy as jnp
import numpy as np
import torch
from model.sharded_falcon3 import FlaxFalcon3ForCausalLM
from test.test_utils import compare_results
from model.convert_hf_weights import torch_to_jnp

def generate(
    model: FlaxFalcon3ForCausalLM = None,
    params: dict = None,
    input_ids: jax.Array = None,
    attention_mask: jax.Array = None,
    position_ids: jax.Array = None,
    max_new_tokens = 20,
    pad_token_id = None,
    eos_token_id = None,
):
    """Generate text efficiently using properly initialized KV caching in NN."""
    # Set defaults for special tokens
    pad_token_id = pad_token_id if pad_token_id is not None else 11
    eos_token_id = eos_token_id if eos_token_id is not None else 11
    
    # Initialize generation variables
    batch_size, seq_length = input_ids.shape

    print(f"Initial pass for loading cache into model...")
    outputs, cache = model.apply(
        params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids[:, :seq_length],
        return_dict=True,
        #init_cache=True,
        mutable=['cache'],
    )
 
    # Get cache, logits and predict next token
    next_token_logits = outputs['logits'][:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    next_token = next_token[:, None]  # Add sequence dimension
    all_token_ids = jnp.concatenate([input_ids, next_token], axis=1)
    params['cache'] = cache['cache']
    
    print(f"Auto-regressive generation to predict next tokens...")
    # Start auto-regressive generation loop
    for i in range(1, max_new_tokens):
        print("------", i, "------") #just to track tokens
        
        outputs, cache = model.apply(
            params,
            input_ids=next_token,  # Only process the new token
            attention_mask=attention_mask,
            position_ids = position_ids[:, seq_length].reshape(-1, 1),
            return_dict=True,
            mutable=['cache'],
        )
        # Get cache, logits and predict next token
        next_token_logits = outputs['logits'][:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token = next_token[:, None]  # Add sequence dimension
        all_token_ids = jnp.concatenate([all_token_ids, next_token], axis=1)
        params['cache'] = cache['cache']

        seq_length += 1
    
    return all_token_ids

jit_generate = jax.jit(
    generate,
    static_argnames=(
        'model',
        'max_new_tokens',
        'pad_token_id',
        'eos_token_id',
    ),
)
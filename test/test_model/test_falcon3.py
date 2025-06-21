import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from model.model_falcon3 import FlaxFalcon3ForCausalLM
from model.configuration_falcon3 import Falcon3Config

def test_falcon3_generation():
    """
    Test FlaxFalcon3ForCausalLM with:
    1. Full forward pass on input prompt
    2. Incremental generation maintaining past key values
    """
    
    config = Falcon3Config(
        num_hidden_layers=2,
    )
    model = FlaxFalcon3ForCausalLM(config)
    
    # Sample prompt
    prompt_text = "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?\n"
    "A: "
    
    # Simulate tokenization (using random tokens for this example)
    prompt_tokens = jnp.array([[1, 464, 2936, 17354, 4419]])  # Shape: (batch_size=1, seq_len=5)
    batch_size, prompt_length = prompt_tokens.shape
    
    print(f"Initial prompt shape: {prompt_tokens.shape}")
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Create attention mask for prompt
    attention_mask = jnp.ones_like(prompt_tokens, dtype=jnp.int32)
    
    # Initialize model parameters
    key = jax.random.PRNGKey(42)
    params = model.init(
        key,
        input_ids=prompt_tokens,
        attention_mask=attention_mask,
        return_dict=True
    )
    
    print("\n" + "="*50)
    print("PHASE 1: Full Forward Pass on Prompt")
    print("="*50)
    
    # Forward pass on full prompt
    prompt_outputs = model.apply(
        params,
        input_ids=prompt_tokens,
        attention_mask=attention_mask,
        return_dict=True,
        output_attentions=True
    )
    
    print(f"Prompt logits shape: {prompt_outputs['logits'].shape}")
    print(f"Last token logits (first 10): {prompt_outputs['logits'][0, -1, :10]}")
    
    # Get next token prediction from prompt
    next_token_logits = prompt_outputs['logits'][0, -1, :]  # Last position logits
    next_token = jnp.argmax(next_token_logits)
    print(f"Next predicted token: {next_token}")
    
    print("\n" + "="*50)
    print("PHASE 2: Incremental Generation with KV Cache")
    print("="*50)
    
    # Set up for generation
    max_new_tokens = 10
    max_length = prompt_length + max_new_tokens
    
    # Prepare inputs for generation (this initializes the cache)
    generation_inputs = model.prepare_inputs_for_generation(
        input_ids=prompt_tokens,
        max_length=max_length,
        attention_mask=attention_mask
    )
    
    print(f"Extended attention mask shape: {generation_inputs['attention_mask'].shape}")
    print(f"Position IDs shape: {generation_inputs['position_ids'].shape}")
    
    # Initialize past key values with full prompt
    initial_outputs = model.apply(
        params,
        input_ids=prompt_tokens,
        attention_mask=generation_inputs['attention_mask'][:, :prompt_length],
        position_ids=generation_inputs['position_ids'][:, :prompt_length],
        past_key_values=generation_inputs['past_key_values'],
        return_dict=True,
        mutable=['cache']
    )
    
    # Extract the updated cache
    past_key_values = initial_outputs[1]['cache'] if isinstance(initial_outputs, tuple) else initial_outputs['past_key_values']
    current_logits = initial_outputs[0]['logits'] if isinstance(initial_outputs, tuple) else initial_outputs['logits']
    
    print(f"Initial cache established. Logits shape: {current_logits.shape}")
    
    # Generated sequence (start with prompt)
    generated_tokens = prompt_tokens.tolist()[0]  # Convert to list for easier manipulation
    current_position = prompt_length
    
    print(f"\nStarting generation from position {current_position}")
    print(f"Initial sequence: {generated_tokens}")
    
    # Generation loop
    for step in range(max_new_tokens):
        print(f"\n--- Generation Step {step + 1} ---")
        
        # Get next token from current logits
        next_token_logits = current_logits[0, -1, :]  # Last position
        next_token = int(jnp.argmax(next_token_logits))
        
        print(f"Generated token: {next_token}")
        print(f"Token probability: {jnp.max(jax.nn.softmax(next_token_logits)):.4f}")
        
        # Add token to sequence
        generated_tokens.append(next_token)
        
        # Prepare next input (single token)
        next_input = jnp.array([[next_token]])  # Shape: (1, 1)
        
        # Update position IDs and attention mask
        current_position += 1
        position_ids = jnp.array([[current_position - 1]])  # Position for the new token
        attention_mask_slice = jnp.ones((1, 1), dtype=jnp.int32)
        
        print(f"Next input shape: {next_input.shape}")
        print(f"Position ID: {position_ids}")
        print(f"Current sequence length: {len(generated_tokens)}")
        
        # Forward pass with cached key-values
        try:
            outputs = model.apply(
                params,
                input_ids=next_input,
                attention_mask=attention_mask_slice,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_dict=True,
                mutable=['cache']
            )
            
            # Update cache and logits
            if isinstance(outputs, tuple):
                current_logits = outputs[0]['logits']
                past_key_values = outputs[1]['cache']
            else:
                current_logits = outputs['logits']
                past_key_values = outputs['past_key_values']
                
            print(f"Output logits shape: {current_logits.shape}")
            
        except Exception as e:
            print(f"Error during generation step {step + 1}: {e}")
            break
            
        # Early stopping condition (you can add custom logic here)
        if next_token == 2:  # Assuming 2 is EOS token
            print("EOS token generated. Stopping.")
            break
    
    print("\n" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(f"Final generated sequence: {generated_tokens}")
    print(f"Total length: {len(generated_tokens)}")
    print(f"New tokens generated: {len(generated_tokens) - prompt_length}")
    
    return generated_tokens

def test_cache_consistency():
    """Test that cached generation produces same results as non-cached for first few tokens"""
    print("\n" + "="*50)
    print("TESTING CACHE CONSISTENCY")
    print("="*50)
    
    config = Falcon3Config(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    
    model = FlaxFalcon3ForCausalLM(config)
    key = jax.random.PRNGKey(123)
    
    # Test sequence
    test_sequence = jnp.array([[1, 2, 3, 4, 5]])
    
    # Initialize params
    params = model.init(key, input_ids=test_sequence[:, :1], return_dict=True)
    
    # Method 1: Process full sequence at once
    full_outputs = model.apply(
        params,
        input_ids=test_sequence,
        return_dict=True
    )
    
    print(f"Full sequence logits shape: {full_outputs['logits'].shape}")
    
    # Method 2: Process incrementally with cache
    cache_outputs = []
    past_kv = None
    
    for i in range(test_sequence.shape[1]):
        current_token = test_sequence[:, i:i+1]
        position_ids = jnp.array([[i]])
        
        if past_kv is None:
            # First token
            outputs = model.apply(
                params,
                input_ids=current_token,
                position_ids=position_ids,
                return_dict=True,
                mutable=['cache']
            )
            if isinstance(outputs, tuple):
                logits = outputs[0]['logits']
                past_kv = outputs[1]['cache']
            else:
                logits = outputs['logits']
                past_kv = outputs.get('past_key_values')
        else:
            # Subsequent tokens
            attention_mask = jnp.ones((1, 1), dtype=jnp.int32)
            outputs = model.apply(
                params,
                input_ids=current_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                return_dict=True,
                mutable=['cache']
            )
            if isinstance(outputs, tuple):
                logits = outputs[0]['logits']
                past_kv = outputs[1]['cache']
            else:
                logits = outputs['logits']
                past_kv = outputs.get('past_key_values')
        
        cache_outputs.append(logits)
        print(f"Token {i+1} processed. Logits shape: {logits.shape}")
    
    # Compare results
    for i, cached_logits in enumerate(cache_outputs):
        full_logits = full_outputs['logits'][:, i:i+1, :]
        diff = jnp.abs(cached_logits - full_logits).max()
        print(f"Position {i}: Max difference = {diff:.6f}")
        
        if diff < 1e-5:
            print(f"✓ Position {i}: Cache consistent")
        else:
            print(f"✗ Position {i}: Cache inconsistent (diff={diff})")

if __name__ == "__main__":
    print("Testing Falcon3 Generation with KV Cache")
    generated_sequence = test_falcon3_generation()
    test_cache_consistency()
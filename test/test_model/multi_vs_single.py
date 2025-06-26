import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from model.convert_hf_weights import make_model
from model.generate_model import generate, jit_generate
from model.convert_hf_weights import torch_to_jnp
from model.jax_config import *
from model.model_falcon3 import debug
def debug_parameter_sharding(params, name="Parameters"):
    """Debug function to show parameter sharding information."""
    from flax.traverse_util import flatten_dict
    
    print(f"\n=== {name} Sharding Debug ===")
    flat_params = flatten_dict(params)
    
    for key, param in flat_params.items():
        print(f"\nParameter: {'.'.join(key)}")
        print(f"  Shape: {param.shape}")
        print(f"  Devices: {len(param.devices())} devices")
        print(f"  Device list: {[str(d) for d in param.devices()]}")
        
        if hasattr(param, 'sharding'):
            print(f"  Sharding spec: {param.sharding}")
            print(f"  Is fully replicated: {param.is_fully_replicated}")
        
        # Show visual sharding for key parameters
        if any(layer_name in '.'.join(key) for layer_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']):
            try:
                print(f"  Visual sharding for {'.'.join(key)}:")
                jax.debug.visualize_array_sharding(param)
            except Exception as e:
                print(f"  Could not visualize: {e}")
    
    print("=" * 60)
def init_model(config, torch_model, batch_size, max_len, rule):
    """
    Initialize the Flax model from the PyTorch model.
    """
    flax_model, flax_params = make_model(
        config=config,
        torch_model=torch_model,
        batch_size=batch_size,
        seq_len=max_len,
        rule=rule
    )
    return flax_model, flax_params

def tokenize(model_name, prompt):
    """
    Prepare input for the PyTorch model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = torch_to_jnp(inputs.input_ids)
    attention_mask = torch_to_jnp(inputs.attention_mask)
    return tokenizer, input_ids, attention_mask

def prepare_input(flax_model, input_ids, attention_mask, max_len):
    """
    Prepare input for the Flax model.
    """
    input_ids = torch_to_jnp(input_ids)
    attention_mask = torch_to_jnp(attention_mask)
    input_ids = jnp.repeat(input_ids, 2, axis=0)  # Duplicate for batch size of 2
    attention_mask = jnp.repeat(attention_mask, 2, axis=0)
    inputs = flax_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
    )
    return input_ids, inputs['attention_mask'], inputs['position_ids']

def shard_input(input_ids, attention_mask, position_ids, device_mesh):

    input_ids = with_named_sharding_constraint(input_ids, device_mesh, P('dp', None))
    attention_mask = with_named_sharding_constraint(attention_mask, device_mesh, P('dp', None))
    position_ids = with_named_sharding_constraint(position_ids, device_mesh, P('dp', None))

    return input_ids, attention_mask, position_ids

def run_model(flax_params, flax_model, input_ids, attention_mask, position_ids, max_len):
    """
    Run the Flax model with the given input IDs and attention mask.
    """
    print("✍️ Generating Flax Model output...")
    _, seq_len = input_ids.shape
    generated_ids = generate(
        params=flax_params,
        model=flax_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_len - seq_len,
    )
    return generated_ids

def strip_output(result, prompt: str = "") -> str:
    """
    Strip the output of the model to remove any special tokens or leading text.
    """
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>"):].lstrip()
    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()
    return result

def compare_results(torch_result: str, flax_result: str) -> str:
    print("Model Result:\n", torch_result)
    print("Sharded Model Result:\n", flax_result)
    if torch_result == flax_result:
        return "✅ Outputs match!"
    else:
        return "❌ Outputs do not match!"


def run_test(model_name: str, prompt: str):
    """
    Run the test comparing Hugging Face and Flax models.
    """
    print("🪄  Initializing models...")
    device_mesh = create_device_mesh(dp_size=2, tp_size=4) 
    config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=2,
        torch_dtype=torch.float32,
    )
    partitioning_rules = get_partitioning_rules(config)

    tokenizer, input_ids, attention_mask = tokenize(model_name, prompt)
    
    batch_size, seq_len = input_ids.shape
    max_len = seq_len + 20  
    model, params = init_model(
        config=config,
        torch_model=None,  # No initial torch model, we will convert later
        batch_size=batch_size * 2,
        max_len=max_len,
        rule="hf"
    )
    sharded_params = shard_params(params, partitioning_rules, device_mesh)
    
    print("🔄 Preparing inputs...")
    input_ids, attention_mask, position_ids = prepare_input(
        flax_model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_len=max_len,
    )

    outputs = run_model(
        flax_params=params,
        flax_model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_len=max_len,
    )
    debug_parameter_sharding(sharded_params, "Sharded Parameters")
    print("🔂 Preparing inputs...")
    input_ids, attention_mask, position_ids = shard_input(input_ids, attention_mask, position_ids, device_mesh)
    print("\n1. Input Tensors:")
    debug(input_ids, "input_ids")
    debug(attention_mask, "attention_mask") 
    debug(position_ids, "position_ids")
    sharded_outputs = run_model(
        flax_params=sharded_params,
        flax_model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_len=max_len,
    )
    
    print("1️⃣ Flax model output:", outputs[0], sep='\n')
    print("🔢 Torch model output:", sharded_outputs[0], sep='\n')
    print("🈵 Decoding outputs...")
    result_single = tokenizer.decode(outputs[0], skip_special_tokens=False)
    result_multi = tokenizer.decode(sharded_outputs[0], skip_special_tokens=False)

    print("🔍 Comparing outputs...")
    print(compare_results(result_single, result_multi))


if __name__ == "__main__":
    run_test(
        model_name="tiiuae/Falcon3-7B-Instruct",
        prompt="""
        Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
        She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
        How much in dollars does she make every day at the farmers' market?\n
        A: 
        """
    )

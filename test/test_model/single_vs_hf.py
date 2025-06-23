import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.model_falcon import FalconForCausalLM
from model.configuration_falcon import FalconConfig
from model.convert_hf_weights import make_model
from model.generate_model import generate, jit_generate
from model.convert_hf_weights import torch_to_jnp

def init_torch_model(model_name: str, config):
    """
    Initialize the PyTorch model with the given configuration.
    """
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return torch_model

def init_flax_model(config, torch_model, batch_size, max_len):
    """
    Initialize the Flax model from the PyTorch model.
    """
    flax_model, flax_params = make_model(
        config=config,
        torch_model=torch_model,
        batch_size=batch_size,
        seq_len=max_len
    )
    return flax_model, flax_params

def prepare_torch_input(model_name, prompt):
    """
    Prepare input for the PyTorch model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    return tokenizer, inputs.input_ids, inputs.attention_mask

def prepare_flax_input(flax_model, input_ids, attention_mask, max_len):
    """
    Prepare input for the Flax model.
    """
    input_ids = torch_to_jnp(input_ids)
    attention_mask = torch_to_jnp(attention_mask)
    inputs = flax_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
    )
    return input_ids, inputs['attention_mask'], inputs['position_ids']

def run_torch_model(torch_model, input_ids, attention_mask):
    """
    Run the PyTorch model with the given input IDs and attention mask.
    """
    print("🏢 Generating HF Model output...")
    outputs = torch_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # return_dict_in_generate=True,
        # output_hidden_states=True,
        # output_attentions=True,
        
    )
    return outputs

def run_flax_model(flax_params, flax_model, input_ids, attention_mask, position_ids, max_len):
    """
    Run the Flax model with the given input IDs and attention mask.
    """
    print("✍️ Generating Flax Model output...")
    _, seq_len = input_ids.shape
    generated_ids = jit_generate(
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
    print("HF Result:\n", torch_result)
    print("Flax Result:\n", flax_result)
    if torch_result == flax_result:
        return "✅ Outputs match!"
    else:
        return "❌ Outputs do not match!"



def run_test(model_name: str, prompt: str):
    """
    Run the test comparing Hugging Face and Flax models.
    """
    print("🪄  Initializing models...")
    config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=2,
        torch_dtype=torch.float32,
    )
    config._attn_implementation = "eager"
    config.dtype = torch.float32
    tokenizer, input_ids, attention_mask = prepare_torch_input(model_name, prompt)

    torch_model = init_torch_model(model_name, config)
    torch_output = run_torch_model(torch_model, input_ids, attention_mask)
    max_len = torch_output.shape[1]
    
    flax_model, flax_params = init_flax_model(
        config=config,
        torch_model=torch_model,
        batch_size=input_ids.shape[0],
        max_len=max_len,
    )
    input_ids, attention_mask, position_ids = prepare_flax_input(
        flax_model=flax_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_len=max_len
    )
    print("🔄 Preparing inputs for Flax model...")
    print(f"Attention mask shape: {attention_mask.shape}")
    flax_output = run_flax_model(
        flax_params=flax_params,
        flax_model=flax_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_len=max_len,
    )

    print("📦 Flax model output:", flax_output[0], sep='\n')
    print("📦 Torch model output:", torch_output[0], sep='\n')
    print("🈵 Decoding outputs...")
    torch_result = tokenizer.decode(torch_output[0], skip_special_tokens=False)
    #torch_result = strip_output(torch_result, prompt)
    flax_result = tokenizer.decode(flax_output[0], skip_special_tokens=False)
    #flax_result = strip_output(flax_result, prompt)

    print("🔍 Comparing outputs...")
    print(compare_results(torch_result, flax_result))


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

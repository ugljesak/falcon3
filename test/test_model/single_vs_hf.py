import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.model_falcon import FalconForCausalLM
from model.configuration_falcon import FalconConfig
from model.convert_hf_weights import make_model
from model.generate_model import generate


def main(
    model_name: str = "tiiuae/falcon-7b-instruct",
    prompt: str = (
    "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?\n"
    "A: ")
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Print all special tokens and their IDs
    for token_name in tokenizer.special_tokens_map:
        token_str = tokenizer.special_tokens_map[token_name]
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        print(f"{token_name}: {token_str} -> {token_id}")

    torch_config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers = 2
    )
    #torch_config.layer_norm_epsilon = 1e-5
    #torch_config.hidden_dropout = 0.0
    #torch_config.parallel_attn = True
    #torch_config.num_ln_in_parallel_attn = 2
    #torch_config.new_decoder_architecture = False
    #torch_config.multi_query = True
    torch_config.group_query = False
    #torch_config.use_cache = True
    #torch_config.bias = False
    print(torch_config)
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=torch_config,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    batch_size = 3
    seq_len = 10
    input_ids = inputs.input_ids.to(torch_model.device)
    attention_mask = inputs.attention_mask.to(torch_model.device)
    print("eos_token_id in input_ids:", tokenizer.eos_token_id in input_ids[0])
    #input_ids = torch.randint(torch_config.vocab_size, (8, 10)).long()
    #attention_mask = torch.ones_like(input_ids)
    print("✍️ Generating...")
    outputs = torch_model.generate(
        input_ids,
        attention_mask=attention_mask
    )
    max_len = outputs[0].shape[0]
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>"):].lstrip()

    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()

    
    batch_size, seq_len = input_ids.shape
    input_ids = jnp.array(input_ids)
    attention_mask = jnp.array(attention_mask)
    print(f"batch_size: {batch_size}, seq_len: {seq_len}")
    print(f"attention_mask.shape: {attention_mask.shape}")
    print(f"Torch model class type: {type(torch_model)}")
    flax_model, flax_params = make_model(torch_config, torch_model)
    
    # for i in range(flax_model.config.num_hidden_layers):
    #     flax_model.model.layers[i].attn.cached_key = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
    #     flax_model.model.layers[i].attn.cached_value = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
    #     flax_model.model.layers[i].attn.cache_index = jnp.array(0, dtype = jnp.int32)

    extended_attention_mask = jnp.ones((batch_size, max_len), dtype = "i4")
    extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    # extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask_jax, (0, 0))
    generated_ids = generate(
        params=flax_params,
        model=flax_model,
        input_ids=input_ids,
        attention_mask=extended_attention_mask,
        max_new_tokens=max_len - seq_len 
    )
    print(generated_ids.shape)
    print(generated_ids)
    result_jax = tokenizer.decode(torch.tensor(generated_ids[0]), skip_special_tokens=False)
    if result_jax.startswith("<|begin_of_text|>"):
        result_jax = result_jax[len("<|begin_of_text|>"):].lstrip()
    if result.startswith(prompt):
        result_jax = result_jax[len(prompt):].lstrip()
    print("OutputJax:", result_jax)

    print("\n🧠 OutputHF:\n", result) 
    print(f"Comparation of hf and flax models: {np.array(result_jax) == np.array(result)}")

main()

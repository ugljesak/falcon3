import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama import FlaxLlamaForCausalLM, LlamaConfig
from model.model_falcon import FalconForCausalLM
from model.configuration_falcon import FalconConfig
from model.convert_hf_weights import make_model
from model.generate_model import generate


def main(
    model_name: str = "tiiuae/Falcon3-7B-Instruct",
    prompt: str = (
    "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?\n"
    "A: ")
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    torch_config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers = 2
    )
    # torch_config.layer_norm_epsilon = 1e-5
    # torch_config.hidden_dropout = 0.0
    # torch_config.parallel_attn = True
    # torch_config.num_ln_in_parallel_attn = 2
    # torch_config.new_decoder_architecture = False
    # torch_config.multi_query = True
    torch_config.group_query = False
    print(torch_config)
    # torch_config.use_cache = True
    # torch_config.bias = False
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=torch_config,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    for name, module in torch_model.named_children():
        print(f"{name}: {module}")

    print("Model config:", torch_model.config)
    print("class type:", type(torch_model))
    print("Model code:\n", torch_model)
    inputs = tokenizer(prompt, return_tensors="pt")
    batch_size = 1
    seq_len = 10
    input_ids = inputs.input_ids.to(torch_model.device)
    attention_mask = inputs.attention_mask.to(torch_model.device)

   

if __name__ == "__main__":
    main()
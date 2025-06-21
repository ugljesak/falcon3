import sys
import jax
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import jax.numpy as jnp
import torch
from ..test_utils import compare_results
from model.model_falcon3 import FlaxFalcon3ForCausalLM
from model.configuration_falcon3 import Falcon3Config

def main(
    model_name: str = "tiiuae/Falcon3-7B-Instruct",
    prompt: str = (
    "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?\n"
    "A: ")
):

    torch_config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=2,
    )
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=torch_config,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, torch_config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int32)
    position_ids = torch.arange(seq_len)[None, :].expand(batch_size, seq_len).to(torch.int32)
    torch_outputs = torch_model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    
    input_ids = jnp.array(input_ids.numpy())
    attention_mask = jnp.array(attention_mask.numpy())
    position_ids = jnp.array(position_ids.numpy())
    flax_config = Falcon3Config(num_hidden_layers=2)
    flax_model = FlaxFalcon3ForCausalLM(flax_config)
    flax_params = flax_model.init(
        jax.random.PRNGKey(0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_attentions=True,
    )

    torch_state_dict = torch_model.state_dict()
    def print_dict_keys(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                print(f"{prefix}{k}")
                print_dict_keys(v, prefix + "  ")
        elif hasattr(d, "keys"):
            for k in d.keys():
                print(f"{prefix}{k}|")
    # print("Torch dict:")
    # print_dict_keys(torch_state_dict)
    # print("Flax dict:")
    # print_dict_keys(flax_params)
    
    def torch_to_jnp(tensor):
        """Convert a PyTorch tensor to a JAX array."""
        return jnp.array(tensor.detach().float().cpu().numpy())

    # Copy all parameters from torch model to flax model
    flax_params['params']['model']['embed_tokens']['embedding'] = torch_to_jnp(
        torch_model.model.embed_tokens.weight
    )
    for i in range(flax_config.num_hidden_layers):
        flax_params['params']['model']['layers'][f'{i}']['input_layernorm']['weight'] = torch_to_jnp(
            torch_model.model.layers[i].input_layernorm.weight
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['q_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.q_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['k_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.k_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['v_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.v_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['self_attn']['o_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].self_attn.o_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['post_attention_layernorm']['weight'] = torch_to_jnp(
            torch_model.model.layers[i].post_attention_layernorm.weight
        )
        flax_params['params']['model']['layers'][f'{i}']['mlp']['up_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].mlp.up_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['mlp']['gate_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].mlp.gate_proj.weight.T
        )
        flax_params['params']['model']['layers'][f'{i}']['mlp']['down_proj']['kernel'] = torch_to_jnp(
            torch_model.model.layers[i].mlp.down_proj.weight.T
        )
    flax_params['params']['model']['norm']['weight'] = torch_to_jnp(
        torch_model.model.norm.weight
    )
    flax_params['params']['lm_head']['kernel'] = torch_to_jnp(
        torch_model.lm_head.weight.T
    )

    flax_outputs = flax_model.apply(
        flax_params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_attentions=True,
    )
    print("Torch outputs logits shape:", torch_outputs.logits.shape)
    print("Flax outputs logits shape:", flax_outputs['logits'].shape)

    compare_results(flax_outputs['logits'], jnp.array(torch_outputs.logits.detach().numpy()))
    print("Torch logits dtype:", torch_outputs.logits.dtype)
    print("Flax logits dtype:", flax_outputs['logits'].dtype)
    print(f"PCC Score: {jnp.min(jnp.corrcoef(flax_outputs['logits'].flatten(), jnp.array(torch_outputs.logits.detach().numpy()).flatten()))}")


if __name__ == "__main__":
    main()


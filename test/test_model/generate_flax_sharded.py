import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
from typing import Union, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.sharded_falcon3 import FlaxFalcon3ForCausalLM
from model.jax_config import shard_params, create_device_mesh, with_named_sharding_constraint
from model.jax_config import with_named_sharding_constraint
from . import *

MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
EXAMPLE_PROMPT = """
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?\n
A: """

def init_flax_model(
    config: AutoConfig, 
    batch_size: int,
    max_len: int,
    checkpoint_path: Optional[Union[str, Path]] = None
) -> tuple[FlaxFalcon3ForCausalLM, dict]:
    """
    Initialize the Flax model with its parameters.
    """
    flax_model = FlaxFalcon3ForCausalLM(config)
    flax_params = flax_model.convert_from_hf_weights(
        config=config,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        max_len=max_len
    )
    return flax_model, flax_params


def prepare_flax_input(
    flax_model: FlaxFalcon3ForCausalLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    max_len: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Prepare input for the Flax model.
    """
    input_ids = jnp.array(input_ids.numpy())
    attention_mask = jnp.array(attention_mask.numpy())
    inputs = flax_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
    )
    return input_ids, inputs['attention_mask'], inputs['position_ids']

def run_flax_model(
    flax_params: dict,
    flax_model: FlaxFalcon3ForCausalLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
    max_len: int
    ) -> jax.Array:
    """
    Run the Flax model with the given proper parameters, input IDs, attention mask and positon IDs.
    """
    print("✍️ Generating Flax Model output...")
    _, seq_len = input_ids.shape
    token_ids = flax_model.generate(
        params=flax_params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_len-seq_len,
    )
    return token_ids

def main(model_name: str, prompt: str):
    # Example usage
    config = AutoConfig.from_pretrained(model_name)
    device_mesh = create_device_mesh(2, 4)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    batch_size, seq_len = inputs.input_ids.shape
    max_len = seq_len + 20
    # Just to test the sharding, we will use a batch size of 2
    batch_size = 2
    inputs.input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
    inputs.attention_mask = jnp.repeat(inputs.attention_mask, batch_size, axis=0)

    flax_model, flax_params = init_flax_model(config, batch_size, max_len)

    input_ids, attention_mask, position_ids = prepare_flax_input(
        flax_model,
        inputs.input_ids,
        inputs.attention_mask,
        max_len
    )
    partitioning_rules = flax_model.get_partitioning_rules()
    flax_params = shard_params(flax_params, partitioning_rules, device_mesh)
    input_ids, attention_mask, position_ids = flax_model.shard_inputs(
        device_mesh, input_ids, attention_mask, position_ids
    )
    # or just do it manually
    # input_ids = with_named_sharding_constraint(input_ids, device_mesh, P('dp', None))
    # attention_mask = with_named_sharding_constraint(attention_mask, device_mesh, P('dp', None))
    # position_ids = with_named_sharding_constraint(position_ids, device_mesh, P('dp', None))

    generated_ids = run_flax_model(
        flax_params,
        flax_model,
        input_ids,
        attention_mask,
        position_ids,
        max_len
    )

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("Generated sequence: ", output[0].strip())


if __name__ == "__main__":
    main(
        model_name=MODEL_NAME,
        prompt=EXAMPLE_PROMPT
    )

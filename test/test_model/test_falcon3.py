import jax
import jax.experimental
import jax.experimental.shard_map
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoConfig
from model.sharded_falcon3 import FlaxFalcon3ForCausalLM
from model.configuration_falcon3 import Falcon3Config
from model.convert_hf_weights import make_model
from model.jax_config import *

def main():
    config = AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Instruct",
        num_hidden_layers=2,
    )
    model = FlaxFalcon3ForCausalLM(config)

    batch_size = 2
    seq_len = 12
    flax_model, flax_params = make_model(
        config=config,
        torch_model=None,
        batch_size=batch_size,
        seq_len=seq_len,
        rule='hf'
    )
    device_mesh = create_device_mesh(dp_size=2, tp_size=4)
    print("Device mesh:", device_mesh)
    rules = model.get_partitioning_rules()
    flax_params = shard_params(flax_params, rules, device_mesh)

    input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    input_ids = jax.lax.with_sharding_constraint(input_ids, NamedSharding(device_mesh, P('dp', None)))
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jax.lax.with_sharding_constraint(attention_mask, NamedSharding(device_mesh, P('dp', None)))
    position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    position_ids = jax.lax.with_sharding_constraint(position_ids, NamedSharding(device_mesh, P('dp', None)))
    jax.debug.visualize_array_sharding(input_ids)
    outputs = model.generate(
        flax_params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=20,
        pad_token_id=11,  # Default pad token ID
        eos_token_id=11,  # Default end of sequence token ID
    )


if __name__ == "__main__":
    main()
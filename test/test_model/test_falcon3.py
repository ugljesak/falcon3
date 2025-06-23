import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoConfig
from model.model_falcon3 import FlaxFalcon3ForCausalLM
from model.configuration_falcon3 import Falcon3Config
from model.convert_hf_weights import make_model
from model.jax_config import create_device_mesh, get_sharding_annotations

def main():
    config = AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Instruct")
    config.num_hidden_layers = 2
    model = FlaxFalcon3ForCausalLM(config)

    batch_size = 2
    seq_len = 10
    flax_model, flax_params = make_model(
        config=config,
        torch_model=None,
        batch_size=batch_size,
        seq_len=seq_len,
        rule='hf'
    )
    device_mesh = create_device_mesh(dp_size=2, tp_size=4)

    def apply(flax_params, input_ids, attention_mask, position_ids):
        outputs = nn.with_partitioning(
            flax_model.apply(
                flax_params,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                mutable=['cache'],
            ),
            get_sharding_annotations(config)
        )
        return outputs
        
    sharded_apply = shard_map(
            apply,
            mesh = device_mesh,
            in_specs = (P(), P(), P(), P()), 
            out_specs = (P(), P()),          
            check_rep = False
        )
    outputs = jax.jit(sharded_apply)(
        flax_params,
        jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0),
        
    )
    next_token_logits = outputs['logits'][:, -1, :]
    sorted_logits = jnp.sort(next_token_logits, axis=-1)
    print("Next token logits:", next_token_logits[:, :, -5:])


if __name__ == "__main__":
    main()
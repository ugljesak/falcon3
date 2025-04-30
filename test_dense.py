import jax
import numpy as np
import sys
sys.path.append("transformers/src/transformers/models/falcon")
from transformers.models.falcon.modeling_falcon import (
    FalconForCausalLM, FalconModel, FalconPreTrainedModel, FalconLinear,
)
from transformers.models.falcon.configuration_falcon import (
    FalconConfig, 
)
from model import DenseLayer
import jax.numpy as jnp
import torch
from utils import compare_results

batch_size = 8
input_dim = 128
output_dim = 128
# Function to create a simple JAX dense layer


torch_linears = [FalconLinear(input_dim, output_dim) for i in range(5)]
jax_dense_layers = [DenseLayer(input_dim, output_dim, use_bias=False) for i in range(5)]
jax_x = jnp.array(np.random.normal(size=(batch_size, input_dim)), dtype=jnp.float32)
torch_x = torch.tensor(np.array(jax_x), dtype=torch.float32)
for torch_linear in torch_linears:
    torch_linear.bias = None 


# params = jax_dense_layer.init(jax.random.PRNGKey(0), jax_x)
params = [jax_dense_layer.init(jax.random.PRNGKey(0), jax_x) for jax_dense_layer in jax_dense_layers]
weights = [param['params']['kernel'] for param in params]
for i in range(len(torch_linears)):
    torch_linears[i].weight.data = torch.nn.Parameter(torch.tensor(weights[i], dtype=torch.float32))
jax_dense_output = jax_x
torch_linear_output = torch_x
for jax_dense_layer, param in zip(jax_dense_layers, params):
    jax_dense_output = jax_dense_layer.apply(param, jax_dense_output)
for torch_linear in torch_linears:
    torch_linear_output = torch_linear(torch_linear_output)
torch_linear_output = torch_linear_output.detach().numpy()

print(f"Dense layer output: {jax_dense_output}")
print(f"Linear layer output: {torch_linear_output}")

# Check if outputs are the same
is_same = jnp.allclose(jax_dense_output, torch_linear_output, atol=1e-5)
compare_results(jax_dense_output, torch_linear_output)
print(f"Dense and linear layers produce the same output: {is_same}")

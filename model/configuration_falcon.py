from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

from transformers import PretrainedConfig
from jax import numpy as jnp

@dataclass
class FalconConfig(PretrainedConfig):
    """Configuration class for Falcon model."""
    vocab_size: int = 65024
    hidden_size: int = 4544
    num_hidden_layers: int = 32
    num_attention_heads: int = 71
    num_kv_heads: Optional[int] = None
    max_position_embeddings: int = 2048
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    bos_token_id: int = 11
    eos_token_id: int = 11
    multi_query: bool = True
    group_query: bool = False
    new_decoder_architecture: bool = True
    parallel_attn: bool = True
    bias: bool = False
    num_ln_in_parallel_attn: Optional[int] = None
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    activation: str = "gelu"
    _attn_implementation: str = "eager"
    ffn_hidden_size: Optional[int] = None
    dtype: jnp.dtype = jnp.float32
    # Depricated attributes
    alibi: Optional[bool] = None
    pruned_heads = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__post_init__()

    def __post_init__(self):

        if self.num_kv_heads is None:
            if self.multi_query:
                self.num_kv_heads = 1
            else:
                self.num_kv_heads = self.num_attention_heads
            
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4
            
        if self.new_decoder_architecture:
            # Override these settings when using new_decoder_architecture
            if self.num_ln_in_parallel_attn is None:
                self.num_ln_in_parallel_attn = 2
                
        # Check for valid configurations
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f"Hidden size {self.hidden_size} must be divisible by number of attention heads {self.num_heads}.")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"Number of attention heads {self.num_heads} must be divisible by number of kv heads {self.num_kv_heads}.")
        

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads
    
    @property
    def rotary(self):
        return True
    
    @property
    def num_heads(self):
        return self.num_attention_heads
    



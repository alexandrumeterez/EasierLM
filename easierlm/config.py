from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import jax.numpy as jnp
import jax

class LayerNormType(Enum):
    ln = "ln"
    rms = "rms"

class ActivationType(Enum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"

class InitFnType(Enum):
    mitchell = "mitchell"
    fan_in = "fan_in"

class DataType(Enum):
    bf16 = jnp.bfloat16
    fp32 = jnp.float32
    fp16 = jnp.float16

@dataclass
class ModelConfig:
    # Layer norm params
    norm_eps: float = 1e-8
    norm_type: LayerNormType = LayerNormType.rms
    elementwise_linear: bool = True

    # Architecture params
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    n_kv_heads: int = 12
    mlp_ratio: int = 4
    activation_type: ActivationType = ActivationType.relu
    dtype: DataType = DataType.bf16
    param_dtype: DataType = DataType.bf16
    precision: jax.lax.Precision = jax.lax.Precision.HIGH
    initializer_range: float = 1.0
    use_rope: bool = True
    max_pos_embed: int = 8192 
    normalize_qk: bool=True

    # data size
    vocab_size: int = 32000
    max_seq_len: int = 16
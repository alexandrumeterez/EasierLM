from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class LayerNormType(str, Enum):
    ln = "ln"
    rms = "rms"

class ActivationType(str, Enum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"

class InitFnType(str, Enum):
    mitchell = "mitchell"
    fan_in = "fan_in"

@dataclass
class ModelConfig:
    # Layer norm params
    norm_eps: float = 1e-8
    norm_type: LayerNormType.rms
    elementwise_linear: bool = True

    # Architecture params
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: int = 4
    activation_type: ActivationType.relu
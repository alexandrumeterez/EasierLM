import numpy as np
import jax
import jax.numpy as jnp
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import linen as nn
from .config import (ModelConfig, LayerNormType)

class RMSNorm(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        eps = self.config.norm_eps
        elementwise_linear = self.config.elementwise_linear

        # autocasting
        og_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = (x**2).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + eps)
        x = x.tp(og_dtype)

        # affine params
        if elementwise_linear:
            d = x.shape[-1]
            weight = self.param("weight", nn.initializers.ones, (d,), og_dtype)
            x = x * weight

        return x

class LayerNorm(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        eps = self.config.norm_eps
        elementwise_linear = self.config.elementwise_linear

        # autocasting
        og_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = (x**2).mean(axis=-1, keepdims=True)
        mean = x.mean(axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(variance + eps)
        x = x.tp(og_dtype)

        # affine params
        if elementwise_linear:
            d = x.shape[-1]
            weight = self.param("weight", nn.initializers.ones, (d,), og_dtype)
            x = x * weight

        return x


def swiglu(x: jnp.ndarray) -> jnp.ndarray:
    a, gate = jnp.split(x, 2, axis=-1)
    return nn.silu(left

class EasierLMBlock(nn.Module):

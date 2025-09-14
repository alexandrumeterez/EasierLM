import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention, dot_product_attention_weights
from flax import linen as nn
from .config import (ModelConfig, LayerNormType, DataType, ActivationType)
import einops

def swiglu(x: jnp.ndarray) -> jnp.ndarray:
    x, gate = jnp.split(x, 2, axis=-1)
    return nn.silu(gate) * x

def get_dtype(dtype: DataType):
    if dtype == DataType.bf16:
        return jnp.bfloat16

# copy pasted from easylm
def apply_rotary_emb(
        xq: jnp.ndarray,
        xk: jnp.ndarray,
        position_ids: jnp.ndarray,
        max_pos: int,
        theta: float=10000.0
):
    input_dtype = xq.dtype
    with jax.ensure_compile_time_eval():
        dim = xq.shape[-1]
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(max_pos)
        freqs = jnp.outer(t, freqs).astype(jnp.float32)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        freqs_cis = jnp.complex64(cos + 1j * sin)
    freqs_cis = jnp.take(freqs_cis, position_ids, axis=0)
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(input_dtype), xk_out.astype(input_dtype)


# copy pasted from easylm
class EasierLMAttention(nn.Module):
    config: ModelConfig  # expects: d_model, n_heads, n_kv_heads, max_pos_embed, dtype, param_dtype, precision,
                         #          initializer_range, norm_eps, elementwise_linear, normalize_qk

    def setup(self) -> None:
        if self.config.normalize_qk:
            self.q_norm = nn.RMSNorm(
                epsilon=self.config.norm_eps,
                dtype=self.config.dtype.value,
                param_dtype=self.config.param_dtype.value,
                use_scale=self.config.elementwise_linear,
                reduction_axes=-1,
            )
            self.k_norm = nn.RMSNorm(
                epsilon=self.config.norm_eps,
                dtype=self.config.dtype.value,
                param_dtype=self.config.param_dtype.value,
                use_scale=self.config.elementwise_linear,
                reduction_axes=-1,
            )

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask,   # shape (b, s) with 1 for keep, 0 for pad
        position_ids,     # shape (b, s) or (s,)
    ):
        cfg = self.config
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = cfg.d_model // cfg.n_heads
        n_kv = getattr(cfg, "n_kv_heads", cfg.n_heads)
        assert cfg.n_heads % n_kv == 0, "n_heads must be divisible by n_kv_heads"
        g = cfg.n_heads // n_kv

        proj_init = jax.nn.initializers.normal(
            cfg.initializer_range / np.sqrt(cfg.d_model)
        )

        # Projections
        xq = nn.Dense(
            cfg.n_heads * head_dim,
            dtype=cfg.dtype.value,
            param_dtype=cfg.param_dtype.value,
            use_bias=False,
            kernel_init=proj_init,
            precision=cfg.precision,
            name="q_proj",
        )(hidden_states)

        xk = nn.Dense(
            n_kv * head_dim,
            dtype=cfg.dtype.value,
            param_dtype=cfg.param_dtype.value,
            use_bias=False,
            kernel_init=proj_init,
            precision=cfg.precision,
            name="k_proj",
        )(hidden_states)

        xv = nn.Dense(
            n_kv * head_dim,
            dtype=cfg.dtype.value,
            param_dtype=cfg.param_dtype.value,
            use_bias=False,
            kernel_init=proj_init,
            precision=cfg.precision,
            name="v_proj",
        )(hidden_states)

        # Reshape
        xq = einops.rearrange(xq, "b s (h d) -> b s h d", h=cfg.n_heads)
        xk = einops.rearrange(xk, "b s (h d) -> b s h d", h=n_kv)
        xv = einops.rearrange(xv, "b s (h d) -> b s h d", h=n_kv)

        # GQA: repeat K/V to match n_heads if needed
        if g > 1:
            xk = einops.repeat(xk, "b s hkv d -> b s (hkv g) d", g=g)
            xv = einops.repeat(xv, "b s hkv d -> b s (hkv g) d", g=g)

        # RoPE (your helper signature may differ)
        xq, xk = apply_rotary_emb(
            xq, xk, position_ids, max_pos=cfg.max_pos_embed
        )

        # Optional QK norm (commonly after RoPE)
        if cfg.normalize_qk:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Masks
        q_len, k_len = xq.shape[1], xk.shape[1]
        with jax.ensure_compile_time_eval():
            full_causal_mask = make_causal_mask(
                jnp.ones((1, cfg.max_pos_embed), dtype=bool), dtype=bool
            )  # (1, 1, S, S)
        causal = full_causal_mask[:, :, :q_len, :k_len]  # (1,1,q,k)

        bsz = hidden_states.shape[0]
        causal = jnp.broadcast_to(causal, (bsz,) + causal.shape[1:])  # (b,1,q,k)

        # attention_mask expected as 1 for keep, 0 for pad
        attn_keep = jnp.expand_dims(attention_mask > 0, axis=(-3, -2))  # (b,1,1,s)
        attn_keep = jnp.broadcast_to(attn_keep, causal.shape)            # (b,1,q,k)

        # Combine (boolean AND)
        combined_mask = jnp.logical_and(causal, attn_keep)               # (b,1,q,k)

        # Convert to bias (add -inf where masked out)
        bias = jax.lax.select(
            combined_mask,
            jnp.zeros(combined_mask.shape, dtype=cfg.dtype.value),
            jnp.full(combined_mask.shape, jnp.finfo(cfg.dtype.value).min, dtype=cfg.dtype.value),
        )

        # Attention
        attn_out = dot_product_attention(
            xq, xk, xv,
            bias=bias,
            mask=None,                 # bias already has both causal+padding
            deterministic=True,
            dtype=cfg.dtype.value,
            precision=cfg.precision,
            dropout_rate=0.0,
        )  # (b, q, h, d)

        # Merge heads
        attn_out = einops.rearrange(attn_out, "b q h d -> b q (h d)")

        # Output projection
        attn_out = nn.Dense(
            cfg.d_model,
            dtype=cfg.dtype.value,
            param_dtype=cfg.param_dtype.value,
            use_bias=False,
            kernel_init=proj_init,
            precision=cfg.precision,
            name="o_proj",
        )(attn_out)
        return attn_out


class EasierLMMLP(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        intermediate_size = self.config.d_model * self.config.mlp_ratio 
        x = nn.Dense(intermediate_size, 
                     use_bias=False, 
                     dtype=self.config.dtype.value, 
                     param_dtype=self.config.param_dtype.value, 
                     kernel_init=nn.initializers.normal(self.config.initializer_range / np.sqrt(self.config.d_model)))(x)


        if self.config.activation_type == ActivationType.gelu:
            x = nn.gelu(x)
        elif self.config.activation_type == ActivationType.relu:
            x = nn.relu(x)
        elif self.config.activation_type == ActivationType.swiglu:
            x = swiglu(x)
        
        x = nn.Dense(self.config.d_model, 
                     use_bias=False, 
                     dtype=self.config.dtype.value, 
                     param_dtype=self.config.param_dtype.value, 
                     kernel_init=nn.initializers.normal(self.config.initializer_range / np.sqrt(intermediate_size)))(x)
        
        return x

class EasierLMBlock(nn.Module):
    config: ModelConfig

    def setup(self) -> None:
        if self.config.norm_type == LayerNormType.rms:
            self.attention_norm = nn.RMSNorm(epsilon=self.config.norm_eps, 
                                             dtype=self.config.dtype.value,
                                             param_dtype=self.config.param_dtype.value,
                                             use_scale=self.config.elementwise_linear,
                                             reduction_axes=-1)
            self.ffn_norm = nn.RMSNorm(epsilon=self.config.norm_eps, 
                                             dtype=self.config.dtype.value,
                                             param_dtype=self.config.param_dtype.value,
                                             use_scale=self.config.elementwise_linear,
                                             reduction_axes=-1)
        elif self.config.norm_type == LayerNormType.ln:
            self.attention_norm = nn.LayerNorm(epsilon=self.config.norm_eps, 
                                             dtype=self.config.dtype.value,
                                             param_dtype=self.config.param_dtype.value,
                                             use_scale=self.config.elementwise_linear,
                                             reduction_axes=-1)
            self.ffn_norm = nn.LayerNorm(epsilon=self.config.norm_eps, 
                                             dtype=self.config.dtype.value,
                                             param_dtype=self.config.param_dtype.value,
                                             use_scale=self.config.elementwise_linear,
                                             reduction_axes=-1)
    @nn.compact
    def __call__(self, hidden_states, attention_mask, position_ids):
        attn_outputs = EasierLMAttention(self.config)(self.attention_norm(hidden_states),
                                                        attention_mask,
                                                        position_ids)
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ffn_norm(hidden_states)
        feed_forward_hidden_states = EasierLMMLP(self.config)(feed_forward_input)
        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,)

class EasierLMBlockGroup(nn.Module):
    config: ModelConfig 

    def setup(self) -> None:
        self.blocks = [EasierLMBlock(self.config) for i in range(self.config.n_layers)]
    
    def __call__(self, hidden_states, attention_mask, position_ids):
        for block in self.blocks:
            layer_outputs = block(hidden_states, attention_mask, position_ids)
            hidden_states = layer_outputs[0]
        outputs = (hidden_states)
        return outputs

class EasierLM(nn.Module):
    config: ModelConfig

    def setup(self) -> None:
        self.wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.config.dtype.value,
            param_dtype=self.config.param_dtype.value,
        )
        self.h = EasierLMBlockGroup(self.config)
        self.ln_f = nn.RMSNorm(
                epsilon=self.config.norm_eps,
                dtype=self.config.dtype.value,
                param_dtype=self.config.param_dtype.value,
                use_scale=self.config.elementwise_linear,
                reduction_axes=-1,
            )
        self.lm_head = nn.Dense(self.config.vocab_size,
                                dtype=self.config.dtype.value,
                                param_dtype=self.config.param_dtype.value,
                                use_bias=False,
                                kernel_init=jax.nn.initializers.normal(
                                    stddev=self.config.initializer_range / np.sqrt(self.config.d_model)), 
                                precision=self.config.precision)
    
    def __call__(self, input_ids, attention_mask, position_ids):
        hidden_states = self.wte(input_ids.astype("i4"))
        outputs = self.h(hidden_states, attention_mask, position_ids=position_ids)
        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.lm_head(hidden_states)
        return hidden_states
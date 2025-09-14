from easierlm.config import ModelConfig
from easierlm.model import EasierLM
from easierlm.jax_utils import JaxRNG

import jax
import jax.numpy as jnp
import optax
import time
from flax.training.train_state import TrainState


def make_dummy_batch(vocab_size: int, batch_size: int, seq_length: int):
    # token ids in [0, vocab_size)
    input_ids = (jnp.arange(seq_length) % vocab_size)[None, :].repeat(batch_size, axis=0).astype(jnp.int32)
    # standard 0..T-1 positions
    position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0).astype(jnp.int32)
    # full attention mask (keep = 1)
    attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    return input_ids, position_ids, attention_mask


def main():
    # ----- 1) Build config/model -----
    config = ModelConfig()
    model = EasierLM(config)

    # Heuristics for names some configs use
    vocab_size = int(getattr(config, "vocab_size", 32000))
    seq_length = int(getattr(config, "seq_len", getattr(config, "max_seq_len", 16)))
    batch_size = 2

    input_ids, position_ids, attention_mask = make_dummy_batch(vocab_size, batch_size, seq_length)

    # ----- 2) Optimizer + TrainState skeleton (even if we only forward-pass) -----
    optimizer = optax.adamw(1e-3)

    # RNGs
    key = jax.random.PRNGKey(0)
    key_params, key_dropout, key_apply = jax.random.split(key, 3)

    # ----- 3) Correct Flax init call (jit compiled) -----
    # NOTE: model.init signature is model.init(rngs, *args, **kwargs)
    # If the model doesn't use dropout at init, the extra key is harmless.
    init_fn = jax.jit(
        lambda k_params, k_drop, x, pos, mask: model.init(
            {"params": k_params, "dropout": k_drop},
            input_ids=x,
            position_ids=pos,
            attention_mask=mask,
        )
    )

    t0 = time.time()
    variables = init_fn(key_params, key_dropout, input_ids, position_ids, attention_mask)
    t1 = time.time()
    print(f"[init] compiled+ran in {t1 - t0:.3f}s")

    # Create TrainState (params only; we're not training here)
    state = TrainState.create(params=variables["params"], tx=optimizer, apply_fn=model.apply)

    # ----- 4) JIT-compiled forward pass -----
    # We pass rngs={'dropout': key} for safety; ignored if model has no dropout.
    @jax.jit
    def fwd(params, x, pos, mask, k_drop):
        return model.apply({"params": params}, x, pos, mask, rngs={"dropout": k_drop})

    # First run triggers compilation
    t0 = time.time()
    out = fwd(state.params, input_ids, position_ids, attention_mask, key_apply)
    t1 = time.time()
    print(f"[apply #1] compiled+ran in {t1 - t0:.3f}s; output shape = {out.shape}")

    # Second run should be fast (cached)
    t0 = time.time()
    out2 = fwd(state.params, input_ids, position_ids, attention_mask, jax.random.split(key_apply, 2)[0])
    t1 = time.time()
    print(f"[apply #2] cached run in {t1 - t0:.6f}s")

    # Peek a few numbers
    print("Output sample (first batch row, first 5 positions):")
    print(jnp.asarray(out[0, :5]))


if __name__ == '__main__':
    main()

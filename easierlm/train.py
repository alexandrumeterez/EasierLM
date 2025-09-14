from easierlm.config import ModelConfig, DataConfig
from easierlm.model import EasierLM
from easierlm.jax_utils import JaxRNG
from easierlm.data.loader import TokenSubsetLoader
import jax
import jax.numpy as jnp
import optax
import time
from flax.training.train_state import TrainState
from tqdm import trange
from tqdm import tqdm
import mlxu
from easierlm.optimizers import AdamWOptimizerFactory
from easierlm.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, make_shard_and_gather_fns,
    with_sharding_constraint,
)
import pprint
from jax.experimental.pjit import pjit

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=0,
    optimizer = AdamWOptimizerFactory.get_default_config()
)

def main(argv):
    set_random_seed(FLAGS.seed)
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    loader = TokenSubsetLoader(data_cfg)
    model = EasierLM(model_cfg)

    optimizer, optimizer_info = AdamWOptimizerFactory.get_optimizer(FLAGS.optimizer)
    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((data_cfg.per_device_batch, data_cfg.seq_len), dtype=jnp.int32),
            position_ids=jnp.zeros((data_cfg.per_device_batch, data_cfg.seq_len), dtype=jnp.int32),
            rngs=rng_generator(('params',)),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)


    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        input_tokens, target_tokens, ce_mask = batch
        position_ids = jnp.tile(jnp.arange(input_tokens.shape[-1])[:, ...], reps=(input_tokens.shape[0], 1))


        def loss_fn(params):
            logits = model.apply(
                params, input_tokens, position_ids,
                rngs=rng_generator(('params',)),
            )
            return cross_entropy_loss(
                logits, target_tokens, ce_mask
            )
        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    jitted_init_fn = pjit(init_fn)
    jitted_train_step = pjit(train_step)

    train_state = jitted_init_fn(next_rng())
    start_step = int(jax.device_get(train_state.step))
    sharded_rng = next_rng()


    for epoch in range(1):
        for step, batch in enumerate(loader.epoch(epoch)):
            train_state, sharded_rng, metrics = jitted_train_step(
                train_state, sharded_rng, batch
            )

            if step % 1 == 0:
                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics = jax.device_get(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
            break
if __name__ == '__main__':
    mlxu.run(main)

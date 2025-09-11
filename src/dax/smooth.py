import abc
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from .particle_approximation import ParticleApproximation, TrajectorySamples


__all__ = ['Smoother', 'BootstrapSmoother']


class Smoother(abc.ABC):

    num_samples: int

    def __init__(self, num_samples):
        self.num_samples = num_samples

    @abc.abstractmethod
    def _smooth(self, ssm, pas, ys, us, key):
        """Sample from the smoothed distribution."""
        pass

    @partial(jax.vmap, in_axes=(None, None, None, None, None, 0))
    def _smooth_vectorized(self, ssm, pas, us, ys, key):
        """Sample from the smoothed distribution."""
        return self._smooth(ssm, pas, us, ys, key)

    @eqx.filter_jit
    def smooth(self, ssm, pas, us, ys, key):
        """Sample from the smoothed distribution."""
        keys = jr.split(key, self.num_samples)
        return TrajectorySamples(self._smooth_vectorized(ssm, pas, us, ys, keys))


class BootstrapSmoother(Smoother):
    """A bootstrap smoother."""

    def _smooth(self, ssm, pas, us, _, key):
        """Sample from the smoothed distribution."""

        def f(carry, data_t):
            x_ts, log_w_ts, u = data_t
            x_tp1, key_t = carry
            log_trans_prob = ssm.transition.cross_log_prob(x_tp1, x_ts, u)
            log_v_ts = log_w_ts + log_trans_prob
            key_t, subkey_t = jr.split(key_t)
            x_t = ParticleApproximation(x_ts, log_v_ts).normalize().sample(subkey_t)
            return (x_t, key_t), x_t

        key, subkey = jr.split(key)
        x_T = pas[-1].sample(subkey)
        init = (x_T, key)
        _, trajectory = jax.lax.scan(f, init,
                                    (pas.particles[-2::-1], pas.log_weights[-2::-1], 
                                    us[-1::-1]))
        trajectory = jnp.concatenate([x_T[None, :], trajectory], axis=0)
        return trajectory[::-1]
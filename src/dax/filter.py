"""Basic filtering and smoothing algorithms."""


import abc
from typing import Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


from .ssm import StateSpaceModel
from .particle_approximation import ParticleApproximation, TrajectoryParticleApproximation


__all__ = ['ParticleApproximation', 'Filter', 'BootstrapFilter']


class Filter(abc.ABC):
    num_particles: int

    def __init__(self, num_particles: int):
        self.num_particles = num_particles

    @abc.abstractmethod
    def filter(self, ssm: StateSpaceModel, us: jax.Array, ys: jax.Array, key) -> Tuple[TrajectoryParticleApproximation, jax.Array]:
        """Run the filter.
        
        It should return a particle approximation of the filtering distribution and the log likelihood of the data.
        """
        pass


class BootstrapFilter(Filter):
    """A class that represents a filter."""

    @eqx.filter_jit
    def filter(self, ssm, us, ys, key):
        """Run the filter."""
        def f(carry, uy):
            pa, key, log_L_prev = carry
            u, y = uy
            key, subkey = jr.split(key)
            tilde_pa = pa.resample(subkey)
            keys = jr.split(subkey, self.num_particles)
            x_next = ssm.transition.sample(tilde_pa.particles, u, keys)
            log_w_next = ssm.likelihood.log_prob(y, x_next, u)
            log_L_next = log_L_prev + jax.scipy.special.logsumexp(log_w_next) - jnp.log(self.num_particles)
            pa_next = ParticleApproximation(x_next, log_w_next).normalize()
            return (pa_next, key, log_L_next), pa_next
        
        key, subkey = jr.split(key)
        keys = jr.split(subkey, self.num_particles)
        pa0 = ParticleApproximation(ssm.x0.sample(keys))
        log_L = 0.0
        init = (pa0, key, log_L)
        (_, _, log_L), pas = jax.lax.scan(f, init, (us, ys))
        return TrajectoryParticleApproximation.make_from_init_and_rest(pa0, pas), log_L
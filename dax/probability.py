"""Some basic probability models. """

import equinox as eqx
import abc
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from functools import partial


__all__ = ['ProbabilityDensity', 'DiagonalGaussian']


class ProbabilityDensity(eqx.Module):

    @abc.abstractmethod
    def _log_prob(self, x):
        """Return the log probability of the PDF."""
        pass

    @abc.abstractmethod
    def _sample(self, key):
        """Sample from the PDF."""
        pass

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0))
    def log_prob(self, x):
        return self._log_prob(x)
    
    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0))
    def sample(self, key):
        return self._sample(key)


class DiagonalGaussian(ProbabilityDensity):

    mean: jax.Array
    log_sigma: jax.Array

    @property
    def sigma(self):
        return jnp.exp(self.log_sigma)
    
    def __init__(self, mean, sigma):
        self.mean = jnp.array(mean)
        self.log_sigma = jnp.log(sigma)
    
    def _log_prob(self, x):
        return -0.5 * jnp.sum( ((x - self.mean) / self.sigma) ** 2) - jnp.sum(self.log_sigma)
    
    def _sample(self, key):
        return self.mean + self.sigma * jr.normal(key, shape=self.mean.shape)
    
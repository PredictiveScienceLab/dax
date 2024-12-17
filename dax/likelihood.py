"""Some basic likelihood models. """

import equinox as eqx
import abc
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from functools import partial


__all__ = ['Likelihood', 'GaussianLikelihood', 'ObservationFunction', 'Identity', 'IDENTITY', 'SubIdentity', 'SingleStateSelector']


class Likelihood(eqx.Module):

    @abc.abstractmethod
    def _log_prob(self, y, x, u):
        """Return the log probability of the observations given the state."""
        pass

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, None, 0, None))
    def log_prob(self, y, x, u):
        return self._log_prob(y, x, u)
    
    @abc.abstractmethod
    def _sample(self, x, u, key):
        """Sample from the likelihood function."""
        pass

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0, 0, 0))
    def sample(self, x, u, key):
        return self._sample(x, u, key)


class ObservationFunction(eqx.Module):

    @abc.abstractmethod
    def __call__(self, x, u):
        """Return the mean of the observation."""
        pass


class Identity(ObservationFunction):

    def __call__(self, x, u):
        return x


IDENTITY = Identity()


class SubIdentity(ObservationFunction):
    _indices: jax.Array

    @property
    def indices(self):
        return jax.lax.stop_gradient(self._indices)

    def __init__(self, indices):
        self._indices = jnp.array(indices)
    
    def __call__(self, x, u):
        return x[self.indices]

    
class SingleStateSelector(ObservationFunction):
    _index: jax.Array

    @property
    def index(self):
        return jax.lax.stop_gradient(self._index)

    def __init__(self, index):
        self._index = jnp.array(index)
    
    def __call__(self, x, u):
        return x[self.index]


class GaussianLikelihood(Likelihood):

    observation_function: eqx.Module
    log_sigma: jax.Array

    @property
    def sigma(self):
        return jnp.exp(self.log_sigma)
    
    def __init__(self, sigma, observation_function=IDENTITY):
        self.observation_function = observation_function
        self.log_sigma = jnp.log(sigma)
    
    def _log_prob(self, y, x, u):
        mean = self.observation_function(x, u)
        return -0.5 * jnp.sum( ((y - mean) / self.sigma) ** 2) - jnp.sum(self.log_sigma)
    
    def _sample(self, x, u, key):
        mean = self.observation_function(x, u)
        return mean + self.sigma * jr.normal(key, shape=x.shape)

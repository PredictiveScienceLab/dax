"""Some basic Markov models. """

import equinox as eqx
import abc
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from functools import partial


__all__ = ['TransitionProbability']


class TransitionProbability(eqx.Module):

    @abc.abstractmethod
    def _log_prob(self, x_next, x_prev, u):
        """Return the log probability of transition."""
        pass

    @abc.abstractmethod
    def _sample(self, x_prev, u, key):
        """Generate a sample from the transition probability."""
        pass

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0, 0, None))
    def log_prob(self, x_next, x_prev, u):
        return self._log_prob(x_next, x_prev, u)
    
    @eqx.filter_jit
    @partial(vmap, in_axes=(None, None, 0, None))
    def cross_log_prob(self, x_next, x_prev, u):
        return self._log_prob(x_next, x_prev, u)

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0, None, None))
    @partial(vmap, in_axes=(None, None, 0, None))
    def pairwise_log_prob(self, x_next, x_prev, u):
        return self._log_prob(x_next, x_prev, u)
    
    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0, None, 0))
    def sample(self, x_prev, u, key):
        return self._sample(x_prev, u, key)
    
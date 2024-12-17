import dax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_sde():

    class SimpleSDE(dax.StochasticDifferentialEquation):
        """A simple SDE with a linear drift and a constant diffusion."""

        mu: jax.Array
        log_sigma: jax.Array

        @property
        def sigma(self):
            return jnp.exp(self.log_sigma)

        def __init__(self, mu, sigma):
            super().__init__()
            self.mu = jnp.array(mu)
            self.log_sigma = jnp.log(sigma)

        def drift(self, x, u):
            return self.mu * x

        def diffusion(self, x, u):
            return jnp.array([[self.sigma]])
    
    sde = SimpleSDE(0.1, 0.01)
    print(sde)

    key = jr.PRNGKey(0)
    x0 = jnp.array([0.])
    sol = sde.sample_path(key, 0., 10., x0, dt=1e-2, max_steps=10000)

    transition = dax.EulerMaruyama(sde, dt=1e-2)
    print(transition)
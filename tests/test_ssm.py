import dax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_ssm():
    from dax import DiagonalGaussian
    from dax import StochasticDifferentialEquation, EulerMaruyama
    from dax import GaussianLikelihood, SingleStateSelector

    class SimpleSDE(StochasticDifferentialEquation):
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
    
    ssm = dax.StateSpaceModel(
        DiagonalGaussian(jnp.array([0.]), jnp.array([1.])),
        EulerMaruyama(SimpleSDE(0.1, 0.1), dt=1e-2), 
        GaussianLikelihood(jnp.array([1.]), SingleStateSelector(0)))
    print(ssm)
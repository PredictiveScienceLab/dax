import dax
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_probability():
    p0 = dax.DiagonalGaussian(jnp.array([0., 0.]), jnp.array([1., 1.]))
    print(p0)
    key = jr.PRNGKey(0)
    keys = jr.split(key, 10)
    x0s = p0.sample(keys)
    log_prob = p0.log_prob(x0s)
    assert log_prob.shape == (10,)
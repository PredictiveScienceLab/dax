import dax
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_likelihood():
    p0 = dax.GaussianLikelihood(jnp.array([1., 1.]))
    key = jr.PRNGKey(0)
    y = jnp.array([1., 1.])
    xs = jr.normal(key, shape=(10, 2))
    u = jnp.array([1., 1.])
    mean = p0.observation_function(xs, u)
    assert mean.shape == (10, 2)

    p1 = dax.GaussianLikelihood(jnp.array([1., 1.]), dax.SubIdentity([0]))
    y = jnp.array([1.])
    log_p1 = p1.log_prob(y, xs, u)
    assert log_p1.shape == (10,)

    p2 = dax.GaussianLikelihood(jnp.array([1., 1.]), dax.SingleStateSelector(0))
    y = 1.0
    log_p2 = p2.log_prob(y, xs, u)
    assert log_p2.shape == (10,)
    
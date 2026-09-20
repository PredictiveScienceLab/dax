import dax
import jax.numpy as jnp
import jax.random as jr


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

    p2 = dax.GaussianLikelihood(1.0, dax.SingleStateSelector(0))
    y = 1.0
    log_p2 = p2.log_prob(y, xs, u)
    assert log_p2.shape == (10,)


def test_gaussian_likelihood_log_prob_is_normalized():
    sigma = jnp.array([2.0, 0.5])
    likelihood = dax.GaussianLikelihood(sigma)
    x = jnp.array([0.25, -0.5])
    y = jnp.array([1.25, 0.0])

    standardized = (y - x) / sigma
    expected = -0.5 * jnp.sum(
        standardized ** 2
        + 2.0 * jnp.log(sigma)
        + jnp.log(2.0 * jnp.pi)
    )

    assert jnp.allclose(likelihood._log_prob(y, x, None), expected)


def test_scalar_observation_scale_broadcasts_over_vector_output():
    sigma = 0.75
    likelihood = dax.GaussianLikelihood(sigma)
    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.5, 0.5])

    expected = -0.5 * jnp.sum(
        ((y - x) / sigma) ** 2
        + 2.0 * jnp.log(sigma)
        + jnp.log(2.0 * jnp.pi)
    )

    assert jnp.allclose(likelihood._log_prob(y, x, None), expected)

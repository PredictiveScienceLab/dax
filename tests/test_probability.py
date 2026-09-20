import dax
import jax.numpy as jnp
import jax.random as jr


def test_probability():
    p0 = dax.DiagonalGaussian(jnp.array([0., 0.]), jnp.array([1., 1.]))
    print(p0)
    key = jr.PRNGKey(0)
    keys = jr.split(key, 10)
    x0s = p0.sample(keys)
    log_prob = p0.log_prob(x0s)
    assert log_prob.shape == (10,)


def test_diagonal_gaussian_log_prob_is_normalized():
    mean = jnp.array([0.5, -1.0])
    sigma = jnp.array([2.0, 0.25])
    x = jnp.array([1.5, -0.5])
    density = dax.DiagonalGaussian(mean, sigma)

    standardized = (x - mean) / sigma
    expected = -0.5 * jnp.sum(
        standardized ** 2
        + 2.0 * jnp.log(sigma)
        + jnp.log(2.0 * jnp.pi)
    )

    assert jnp.allclose(density._log_prob(x), expected)

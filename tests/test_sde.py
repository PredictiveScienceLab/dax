import dax
import jax
import jax.numpy as jnp
import jax.random as jr


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
            return jnp.array([self.sigma])
    
    sde = SimpleSDE(0.1, 0.01)
    print(sde)

    key = jr.PRNGKey(0)
    x0 = jnp.array([0.])
    sol = sde.sample_path(key, 0., 10., x0, dt=1e-2, max_steps=10000)

    transition = dax.EulerMaruyama(sde, dt=1e-2)
    print(transition)


def test_euler_maruyama_log_prob_matches_sampling_distribution():
    class ConstantSDE(dax.StochasticDifferentialEquation):
        drift_value: jax.Array
        diffusion_value: jax.Array

        def __init__(self, drift, diffusion):
            super().__init__()
            self.drift_value = jnp.asarray(drift)
            self.diffusion_value = jnp.asarray(diffusion)

        def drift(self, x, u):
            del x, u
            return self.drift_value

        def diffusion(self, x, u):
            del x, u
            return self.diffusion_value

    dt = 0.04
    diffusion = jnp.array([0.5, -0.25])
    transition = dax.EulerMaruyama(
        ConstantSDE(jnp.array([0.2, -0.1]), diffusion), dt=dt
    )
    x_prev = jnp.array([1.0, -2.0])
    mean = x_prev + transition.sde.drift(x_prev, None) * dt
    variance = diffusion ** 2 * dt

    expected_at_mean = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * variance))
    assert jnp.allclose(
        transition._log_prob(mean, x_prev, None), expected_at_mean
    )

    one_standard_deviation = mean + jnp.sqrt(variance)
    assert jnp.allclose(
        transition._log_prob(one_standard_deviation, x_prev, None),
        expected_at_mean - 0.5 * mean.size,
    )

    keys = jr.split(jr.key(17), 20_000)
    x_prev_batch = jnp.broadcast_to(x_prev, (keys.shape[0], x_prev.size))
    samples = transition.sample(x_prev_batch, None, keys)
    assert jnp.allclose(jnp.var(samples, axis=0), variance, rtol=0.08)

import dax
import jax
import jax.numpy as jnp
import jax.random as jr


class BrownianMotion(dax.StochasticDifferentialEquation):
    diffusion_amplitude: jax.Array

    def __init__(self, diffusion_amplitude):
        super().__init__()
        self.diffusion_amplitude = jnp.asarray(diffusion_amplitude)

    def drift(self, x, u):
        del u
        return jnp.zeros_like(x)

    def diffusion(self, x, u):
        del x, u
        return self.diffusion_amplitude


def test_bootstrap_filter_uses_independent_keys():
    ssm = dax.StateSpaceModel(
        dax.DiagonalGaussian(jnp.zeros(1), jnp.ones(1)),
        dax.EulerMaruyama(BrownianMotion(jnp.array([0.2])), dt=0.1),
        dax.GaussianLikelihood(0.5, dax.SingleStateSelector(0)),
    )
    controls = jnp.zeros(4)
    observations = jnp.zeros(4)
    particle_filter = dax.BootstrapFilter(num_particles=32)

    with jax.debug_key_reuse(True):
        particles, log_likelihood = particle_filter.filter(
            ssm, controls, observations, jr.key(0)
        )

    assert particles.particles.shape == (5, 32, 1)
    assert jnp.allclose(jnp.sum(particles.weights, axis=1), 1.0)
    assert jnp.isfinite(log_likelihood)

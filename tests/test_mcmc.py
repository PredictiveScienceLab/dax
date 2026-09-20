import dax
import jax
import jax.numpy as jnp
import jax.random as jr


class StandardNormalPrior(dax.Prior):
    def log_prob(self, theta):
        return -0.5 * theta ** 2

    def sample(self, key):
        return jr.normal(key)


class KeyConsumingFilter(dax.Filter):
    def __init__(self):
        super().__init__(num_particles=1)

    def filter(self, ssm, us, ys, key):
        del us, ys
        random_zero = 0.0 * jr.normal(key)
        return jnp.array(0.0), -0.5 * ssm ** 2 + random_zero


def identity_model(theta):
    return theta


def test_particle_mcmc_separates_initial_filter_and_chain_keys():
    sampler = dax.ParticleMCMC(
        StandardNormalPrior(),
        KeyConsumingFilter(),
        proposal_scale=0.1,
        ssm_from_theta=identity_model,
    )

    with jax.debug_key_reuse(True):
        state, results = sampler.run(
            theta=jnp.array(0.0),
            us=jnp.empty(0),
            ys=jnp.empty(0),
            num_steps=1,
            key=jr.key(0),
        )

    assert jnp.isfinite(state[1])
    assert results[0].shape == (1,)

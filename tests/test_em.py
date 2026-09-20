import dax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


class QuadraticModel(eqx.Module):
    parameter: jax.Array

    def _log_prob(self, xs, us, ys):
        del us, ys
        return -(self.parameter - xs[0, 0]) ** 2


class ParameterDependentSmoother:
    def smooth(self, ssm, pas, us, ys, key):
        del pas, us, ys, key
        return dax.TrajectorySamples(
            jnp.array([[[2.0 * ssm.parameter]]])
        )


def test_m_step_holds_smoothing_draws_fixed():
    em = dax.ExpectationMaximization(
        filter=None,
        smoother=ParameterDependentSmoother(),
        optimizer=optax.sgd(learning_rate=0.25),
        max_m_step_iters=2,
    )
    model = QuadraticModel(jnp.array(1.0))

    updated_model, _, _ = em.m_step(
        model,
        pas=None,
        us=jnp.empty(0),
        ys=jnp.empty(0),
        key=jr.key(0),
    )

    # The single smoothing draw targets 2.0. Two gradient steps from 1.0
    # produce 1.75. Redrawing from the changing model would produce 2.25.
    assert jnp.allclose(updated_model.parameter, 1.75)

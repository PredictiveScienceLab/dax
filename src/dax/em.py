"""Implementation of the expectation-maximization algorithm for particle filters."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


from . import Filter, Smoother


__all__ = ['ExpectationMaximization']


class ExpectationMaximization(eqx.Module):
    """Particle Monte Carlo EM with fixed smoothing draws in each M-step.

    Finite particle approximations and numerical M-steps do not guarantee
    monotonic ascent of the observed-data likelihood.
    """

    filter: Filter
    smoother: Smoother
    optimizer: optax.GradientTransformation
    max_m_step_iters: int

    def __init__(self, filter: Filter, smoother: Smoother, optimizer: optax.GradientTransformation,
                 max_m_step_iters: int = 1):
        self.filter = filter
        self.smoother = smoother
        self.optimizer = optimizer
        self.max_m_step_iters = max_m_step_iters
    
    @eqx.filter_jit
    def e_step(self, ssm, us, ys, key):
        """Run the E-step of the EM algorithm."""
        key, subkey = jax.random.split(key)
        pas, log_L = self.filter.filter(ssm, us, ys, subkey)
        return pas, log_L, key
    
    #@eqx.filter_jit
    def m_step(self, ssm, pas, us, ys, key):
        """Optimize one Monte Carlo objective using fixed smoothing draws."""
        def loss_fn(model, trajectories):
            return -trajectories.expect(lambda xs: model._log_prob(xs, us, ys))

        key, smoothing_key = jax.random.split(key)
        sampled_trajectories = self.smoother.smooth(
            ssm, pas, us, ys, smoothing_key
        )
        sampled_trajectories = jax.tree.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            sampled_trajectories,
        )

        def inner_m_step(i, val):
            _ssm, _opt_state = val
            grads = eqx.filter_grad(loss_fn)(_ssm, sampled_trajectories)
            updates, _opt_state_next = self.optimizer.update(
                grads, _opt_state, eqx.filter(_ssm, eqx.is_array)
            )
            _ssm_next = eqx.apply_updates(_ssm, updates)
            return _ssm_next, _opt_state_next

        opt_state = self.optimizer.init(eqx.filter(ssm, eqx.is_array))
        init = (ssm, opt_state)
        ssm, opt_state = jax.lax.fori_loop(
            0, self.max_m_step_iters, inner_m_step, init
        )
        return ssm, opt_state, key

    @eqx.filter_jit
    def step(self, ssm, us, ys, key):
        """Run a step of the EM algorithm."""
        pas, log_L, key = self.e_step(ssm, us, ys, key)
        ssm, _, key = self.m_step(ssm, pas, us, ys, key)
        return ssm, log_L, key
    
    def run(self, ssm, us, ys, num_steps, key):
        """Run the EM algorithm."""
        log_Ls = []
        for _ in range(num_steps):
            ssm, log_L, key = self.step(ssm, us, ys, key)
            print(f'log likelihood: {log_L:.2e}')
            log_Ls.append(log_L)
        log_Ls = jnp.array(log_Ls)
        return ssm, log_Ls

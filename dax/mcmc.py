"""Implementation of Particle MCMC."""

import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
import abc
import blackjax


from . import Filter


__all__ = ['Prior', 'ParticleMCMC']


class Prior(eqx.Module):

    @abc.abstractmethod
    def log_prob(self, theta):
        """Return the log probability of the prior."""
        pass

    @abc.abstractmethod
    def sample(self, key):
        """Generate a sample from the prior."""
        pass


class ParticleMCMC(eqx.Module):

    prior: Prior
    filter: Filter
    proposal: Callable
    ssm_from_theta: Callable


    def __init__(self, prior: Prior, filter: Filter, proposal_scale: float, ssm_from_theta: Callable):
        self.prior = prior
        self.filter = filter
        self.proposal = blackjax.mcmc.random_walk.normal(proposal_scale)
        self.ssm_from_theta = ssm_from_theta

    @eqx.filter_jit
    def step(self, state):
        """Run a step of the Metropolis-Hastings algorithm."""
        theta_prev, log_L_prev, log_prior_prev, key, us, ys = state
        
        key, subkey = jax.random.split(key)
        dtheta = self.proposal(subkey, theta_prev)
        theta_next = jax.tree_map(lambda x, dx: x + dx, theta_prev, dtheta)
   
        key, subkey = jax.random.split(key)
        ssm_next = self.ssm_from_theta(theta_next)

        log_L_next = self.filter.filter(ssm_next, us, ys, subkey)[1]

        log_prior_next = self.prior.log_prob(theta_next)
        log_alpha = log_L_next + log_prior_next - log_L_prev - log_prior_prev
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        #print(f"log_L_prev: {log_L_prev:0.2f}, log_L_next: {log_L_next:0.2f}, ar = {jnp.exp(log_alpha):0.2f}, alpha: {theta_next['alpha']:0.2f}")
        accept = jax.lax.lt(jnp.log(u), log_alpha)
        theta_next, log_L_next, log_prior_next = jax.lax.cond(
            accept,
            lambda _: (theta_next, log_L_next, log_prior_next),
            lambda _: (theta_prev, log_L_prev, log_prior_prev),
            None
        )
        return (theta_next, log_L_next, log_prior_next, key, us, ys), (accept, log_L_next, theta_next)
    
    @eqx.filter_jit
    def run(self, theta, us, ys, num_steps, key):
        """Run the Particle MCMC algorithm."""
        log_prior = self.prior.log_prob(theta)
        ssm = self.ssm_from_theta(theta)
        _, log_L = self.filter.filter(ssm, us, ys, key)
        state = (theta, log_L, log_prior, key, us, ys)
        state, results = jax.lax.scan(lambda state, _: self.step(state), state, jnp.arange(num_steps))
        return state, results

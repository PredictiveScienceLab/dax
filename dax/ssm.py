"""Classes and methods about state space models."""

__all__ = ['StateSpaceModel']

import equinox as eqx
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from functools import partial


from .probability import ProbabilityDensity
from .markov import TransitionProbability
from .likelihood import Likelihood
from .particle_approximation import ParticleApproximation, TrajectoryParticleApproximation


class StateSpaceModel(eqx.Module):
    """A class that represents a state space model."""

    x0: ProbabilityDensity
    transition: TransitionProbability
    likelihood: Likelihood

    def __init__(self, x0, transition, likelihood):
        self.x0 = x0
        self.transition = transition
        self.likelihood = likelihood

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0, None, None))
    def log_prob(self, xs, us, ys):
        return self._log_prob(xs, us, ys)

    def _log_prob(self, xs, us, ys):
        
        def f(carry, xuy):
            x_prev = carry
            x_next, u_prev, y_next = xuy
            log_prob_t = self.transition._log_prob(x_next, x_prev, u_prev)
            log_prob_o = self.likelihood._log_prob(y_next, x_next, u_prev)
            return x_next, log_prob_t + log_prob_o
        
        log_prob_init = self.x0._log_prob(xs[0])
        init = xs[0]
        _, log_prob_ts = jax.lax.scan(f, init, (xs[1:], us, ys))
        return log_prob_init + jnp.sum(log_prob_ts, axis=0)
    

    @eqx.filter_jit
    def predict(self, pa0, us_next, key):
        def f(carry, u):
            pa, key = carry
            key, subkey = jr.split(key)
            keys = jr.split(subkey, pa.num_particles)
            pa_next = ParticleApproximation(self.transition.sample(pa.particles, u, keys))
            return (pa_next, key), pa
        
        key, subkey = jr.split(key)
        init = (pa0.resample(subkey), key)
        _, pas = jax.lax.scan(f, init, us_next)
        return TrajectoryParticleApproximation.make_from_particle_approximation(pas, resampled=True)
    

if __name__ == '__main__':
    from probability import DiagonalGaussian
    from sde import StochasticDifferentialEquation, EulerMaruyama
    from likelihood import GaussianLikelihood, SingleStateSelector

    class SimpleSDE(StochasticDifferentialEquation):
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
            return jnp.array([[self.sigma]])
    
    ssm = StateSpaceModel(
        DiagonalGaussian(jnp.array([0.]), jnp.array([1.])),
        EulerMaruyama(SimpleSDE(0.1, 0.1), dt=1e-2), 
        GaussianLikelihood(jnp.array([1.]), SingleStateSelector(0)))
    print(ssm)
"""Some material on SDEs."""

import jax
from jax import vmap
import jax.random as jr
import jax.numpy as jnp
import abc
import equinox as eqx
from functools import partial

from .markov import TransitionProbability


__all__ = ['ControlFunction', 'ZeroControl', 'TIME_CONTROL', 'StochasticDifferentialEquation', 'EulerMaruyama']


class ControlFunction(eqx.Module):

    @abc.abstractmethod
    def _eval(self, t):
        """Return the control at time t."""
        pass

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0))
    def __call__(self, t):
        return self._eval(t)


class ZeroControl(ControlFunction):

    def _eval(self, t):
        return 0.
    
ZERO_CONTROL = ZeroControl()


class TimeControl(ControlFunction):

    def _eval(self, t):
        return t

TIME_CONTROL = TimeControl()


def times_between(t0, t1, dt):
    return jnp.linspace(t0, t1, int((t1 - t0) / dt) + 1)


class StochasticDifferentialEquation(eqx.Module):
    """A class that represents a stochastic differential equation (SDE).

    The SDE is described in the Itô sence:

        dX_t = drift(X_t, u_t)dt + diffusion(X_t, u_t) @ dW_t,

    where:

        - X_t is d-dimensional (d is unspecified)
        - u_t is an arbitrary vector
        - drift(X_t, u_t) is d-dimensional
        - diffusion(X_t, u_t) is a diagonal matrix
        - W_t is a d-dimensional Wiener measure

    """

    control_function: eqx.Module

    def __init__(self, control_function=ZERO_CONTROL):
        self.control_function = control_function
    
    @abc.abstractmethod
    def drift(self, x, u):
        """The drift term of the SDE."""
        pass

    @abc.abstractmethod
    def diffusion(self, x, u):
        """The diffusion term of the SDE."""
        pass

    @eqx.filter_jit
    def sample_path(self, key, t0, t1, x0, dt=1e-3, dt0=1e-3, **kwargs):
        """Sample a path from the SDE."""
        from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
        drift = lambda t, x, args: self.drift(x, self.control_function._eval(t))
        diffusion = lambda t, x, args: self.diffusion(x, self.control_function._eval(t))
        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(x0.shape[0],), key=key)
        terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
        solver = Euler()
        ts = times_between(t0, t1, dt)
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=x0, saveat=saveat, **kwargs)
        return sol
    
 
class EulerMaruyama(TransitionProbability):
    sde: StochasticDifferentialEquation
    _dt: jax.Array

    def ts(self, t0, t1):
        return times_between(t0, t1, self.dt)

    @property
    def dt(self):
        return jax.lax.stop_gradient(self._dt)

    def __init__(self, sde, dt=1e-1):
        self.sde = sde
        self._dt = jnp.array(dt)

    def _log_prob(self, x_next, x_prev, u):
        drift = self.sde.drift(x_prev, u)
        diffusion = self.sde.diffusion(x_prev, u)
        return jnp.sum(-0.5 * ((x_next - x_prev - drift * self.dt) / diffusion) ** 2) - jnp.sum(jnp.log(diffusion))
    
    def _sample(self, x_prev, u, key):
        drift = self.sde.drift(x_prev, u)
        diffusion = self.sde.diffusion(x_prev, u)
        return x_prev + drift * self.dt + diffusion * jr.normal(key, shape=x_prev.shape) * jnp.sqrt(self.dt)

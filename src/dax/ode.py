import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import abc

from .control import ZERO_CONTROL

class OrdinaryDifferentialEquation(eqx.Module):
    """A class that represents an ordinary differential equation (ODE).
    
    The ODE is described by:

        dx_t = vector_field(x_t, u_t)dt,

    where:
        - x_t is d-dimensional (d is unspecified)
        - u_t is an arbitrary vector
        - vector_field(x_t, u_t) is d-dimensional
    """
    control_function: eqx.Module

    def __init__(self, control_function=ZERO_CONTROL):
        self.control_function = control_function
    
    @abc.abstractmethod
    def vector_field(self, x, u):
        """The vector field of the ODE."""
        pass

    @eqx.filter_jit
    def solve(self, x0, times, solver=Tsit5(), dt0=1e-3, **kwargs):
        vector_field = lambda t, x, args: self.vector_field(x, self.control_function._eval(t))
        term = ODETerm(vector_field)
        saveat = SaveAt(ts=times)
        sol = diffeqsolve(term, solver, t0=times[0], t1=times[-1], dt0=dt0, y0=x0, saveat=saveat, **kwargs)
        return sol
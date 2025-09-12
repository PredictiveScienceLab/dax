from jax import vmap
import abc
import equinox as eqx
from functools import partial

__all__ = ['ControlFunction', 'ZeroControl', 'TimeControl', 'ZERO_CONTROL', 'TIME_CONTROL']

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

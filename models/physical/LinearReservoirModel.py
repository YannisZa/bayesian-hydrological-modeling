from theano.compile.ops import as_op
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import theano.tensor as tt
from theano import *
import theano
import numpy as np
import pymc3 as pm
import pandas as pd

theano.config.exception_verbosity='high'

class LinearReservoirModel(object):

    def __init__(self, times, nrint, q0=0.1):
            self._q0 = np.array([q0], dtype=np.float64)
            self._times = times
            self._nrint = nrint

    def _simulate(self, parameters, times):
        k = float(parameters[0])

        def rhs(y, t, p):
            return (self._nrint(t) - y) * (1./k)

        values = odeint(rhs, self._q0, times, (parameters,),rtol=1e-6,atol=1e-6)
        return values

    def simulate(self, x):
        return self._simulate(x, self._times)

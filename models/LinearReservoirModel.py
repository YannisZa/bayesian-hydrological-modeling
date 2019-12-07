from scipy.interpolate import interp1d
from scipy.integrate import odeint
import theano
from theano import *
from theano.compile.ops import as_op
import theano.tensor as tt
import numpy as np
import pymc3 as pm
import pandas as pd
theano.config.exception_verbosity='high'

class LinearReservoirModel(object):

    def __init__(self, net_rainfall_data, params):
            self._q0 = np.array([params.q0], dtype=np.float64)
            self._times = range(0,len(net_rainfall_data))
            self._nrint = interp1d(self._times, net_rainfall_data,fill_value="extrapolate",kind='slinear')
            self._n_times = len(net_rainfall_data)
            self._n_states = 1

    def _simulate(self, parameters, times, factor):
        k = float(parameters[0])

        def rhs(y, t, p):
            return (max(self._nrint(t),0.0001) - y) * (1./k)
            # return (self._nrint(t) - y) * (1./k)

        values = odeint(rhs, self._q0, self._times, (parameters,),rtol=1e-6,atol=1e-6)
        nonzero_values = np.array([[max(0,qsim[0])*factor] for qsim in values]).reshape(len(values),1)

        # return values
        return nonzero_values

    def simulate(self, x, factor):
        return self._simulate(x, self._times, factor)

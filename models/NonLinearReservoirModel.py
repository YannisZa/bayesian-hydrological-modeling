from scipy.interpolate import interp1d
from scipy.integrate import odeint
import theano
from theano import *
import theano.tensor as tt
import numpy as np
import pymc3 as pm
import pandas as pd
theano.config.exception_verbosity='high'

class NonLinearReservoirModel(object):

    def __init__(self, net_rainfall_data, params):
            self._q0 = np.array([params.q0], dtype=np.float64)
            self._times = range(0,len(net_rainfall_data))
            self._nrint = interp1d(self._times, net_rainfall_data,fill_value="extrapolate",kind='slinear')
            self._n_times = len(net_rainfall_data)
            self._n_states = 1

    def _simulate(self, parameters, times):
        k,m = [float(x) for x in parameters]

        a = m*(1./k)**(1./m)
        b = (m-1)/m

        def rhs(y, t, p):
            return ( a * ( nrint(time) - y) * (y**b) )

        values = odeint(rhs, self._q0, self._times, (parameters,),rtol=1e-6,atol=1e-6)
        return values

    def simulate(self, x):
        return self._simulate(x, self._times)

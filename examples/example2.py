import theano
from theano import *
import theano.tensor as tt
from theano.compile.ops import as_op
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import pymc3 as pm
theano.config.exception_verbosity='high'

class FitzhughNagumoModel(object):
    def __init__(self, times, y0=None):
            self._y0 = np.array([-1, 1], dtype=np.float64)
            self._times = times

    def _simulate(self, parameters, times):
        a, b, c = [float(x) for x in parameters]

        def rhs(y, t, p):
            V, R = y
            dV_dt = (V - V**3 / 3 + R) * c
            dR_dt = (V - a + b * R) / -c
            return dV_dt, dR_dt
        values = odeint(rhs, self._y0, times, (parameters,),rtol=1e-6,atol=1e-6)
        return values

    def simulate(self, x):
        return self._simulate(x, self._times)

n_states = 2
n_times = 200
true_params = [0.2,0.2,3.]
noise_sigma = 0.5
FN_solver_times = np.linspace(0, 20, n_times)
ode_model = FitzhughNagumoModel(FN_solver_times)
sim_data = ode_model.simulate(true_params)
np.random.seed(42)
Y_sim = sim_data + np.random.randn(n_times,n_states)*noise_sigma

@as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(param1,param2,param3):

    param = [param1,param2,param3]
    th_states = ode_model.simulate(param)

    return th_states


n_chains = 100               # WARNING: Always use n_chains >= 500 for real applications

with pm.Model() as FN_model:

    a = pm.Gamma('a', alpha=2, beta=1)
    b = pm.Normal('b', mu=0, sd=1)
    c = pm.Uniform('c', lower=0.1, upper=10)

    sigma = pm.HalfNormal('sigma', sd=1)

    forward = th_forward_model(a,b,c)

    cov=np.eye(2)*sigma**2

    Y_obs = pm.MvNormal('Y_obs', mu=forward, cov=cov, observed=Y_sim)

    # tt.printing.Print('Y_obs')(Y_obs)
    # print(Y_sim)

    # Initial points for each of the chains
    np.random.seed(21)
    startsmc=[{'a':np.random.uniform(1e-3,2,1),'b':np.random.uniform(1e-3,2,1),
            'c':np.random.uniform(1e-3,10,1),'sigma':np.random.uniform(1e-3,2)} for _ in range(n_chains)]

    trace_FN = pm.sample(1000, progressbar=True, chains=n_chains, start=startsmc, step=pm.SMC())

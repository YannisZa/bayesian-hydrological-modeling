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


q = [0.01, 0.084788051,0.289827287,0.487426902,0.623592162,0.855202214,0.901709887,0.87936577,0.857067839,0.775516564,0.701725939,0.675138958,0.68101658,0.64644605,0.701305112,0.747128907,0.676039744,0.668502137,0.731464651,0.766588801]
nr = [1.618666063,0.0001,4.405308823,0.394073731,3.392555321,2.733285785,0.0001,1.31186209,0.0001,0.0001,0.0001,0.83074128,0.646141131,0.0001,2.405660466,0.0001,0.0001,1.174002978,1.481146447,0.73244669]

times = range(0,len(q))

nr_int = interp1d(times, nr,fill_value="extrapolate",kind='slinear')

q0 = 0.2
n_states = 1
n_times = 20
true_params = [5]
noise_sigma = 0.01
ode_model = LinearReservoirModel(times,nr_int,q0)
sim_data = ode_model.simulate(true_params)
np.random.seed(42)

Q_sim = sim_data + np.random.randn(n_times,n_states)*noise_sigma

@as_op(itypes=[tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(param1):
    params = [param1]

    th_states = ode_model.simulate(params)
    return th_states

n_chains = 500 # WARNING: Always use n_chains >= 500 for real applications
kmax = 10
#musigma = 1
sdsigma = 0.01

# Define the data matrix
Q = np.vstack((q))

with pm.Model() as LR_model:

    # Priors for unknown model parameters
    k = pm.Uniform('k', lower=0.01, upper=kmax)
    # k = pm.Lognormal('k', mu=params.kmax, sd=1)

    # Priors for initial conditions and noise level
    #q0 = pm.Lognormal('q0', mu=muq, sd=sdq)
    sigma = pm.HalfNormal('sigma', sd=sdsigma)

    forward = th_forward_model(k)

    Q_obs = pm.Lognormal('Q_obs', mu=pm.math.log(forward), sigma=sigma, observed=Q_sim)

    #tt.printing.Print('Q_obs')(Q_obs)
    # print(Q_sim)


    # Initial points for each of the chains
    np.random.seed(21)
    startsmc = [{'k':np.random.uniform(1e-3,kmax,1)} for _ in range(n_chains)]

    trace_LR = pm.sample(1000, progressbar=True, chains=n_chains, start=startsmc, step=pm.SMC())



# results=[pm.summary(trace_LR, ['k'])]
# results=pd.concat(results)
# true_params.append(noise_sigma)
# results['True values'] = pd.Series(np.array(true_params), index=results.index)
# true_params.pop();
# print(results)

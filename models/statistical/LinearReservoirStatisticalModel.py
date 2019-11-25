from scipy.interpolate import interp1d
from scipy.integrate import odeint
from models.physical.LinearReservoirModel import LinearReservoirModel as LRM
import theano.tensor as tt
import theano
import numpy as np
import pymc3 as pm
import math
from theano.compile.ops import as_op
from theano import *

theano.config.exception_verbosity= 'high'
theano.config.floatX = 'float64'
print('Running on PyMC3 v{}'.format(pm.__version__))

n_states = 1
n_odeparams = 1
n_ivs = 1

class LinearReservoirStatisticalModel(object):

    def __init__(self, net_rainfall_data, params, seed=42):
        self._times = range(0,len(net_rainfall_data))
        self._nrint = interp1d(self._times, net_rainfall_data,fill_value="extrapolate",kind='slinear')
        self._n_states = 1
        self._n_times = len(net_rainfall_data)
        self._ode_model = LRM(self._times,self._nrint,params.q0)

    def _simulate_q(self, params):

        # Simulate Q data
        sim_data = self._ode_model.simulate([params.k])
        # Fix random seed
        np.random.seed(42)
        # Add noise to Q
        Q_sim = sim_data + np.random.randn(self._n_times,self._n_states)*params.sigma

        return Q_sim


    def _sample(self,q,Q_sim,params):

        @as_op(itypes=[tt.dscalar], otypes=[tt.dmatrix])
        def th_forward_lrmodel(param1):
            parameter_list = [param1]

            th_states = self._ode_model.simulate(parameter_list)
            return th_states

        # Define the data matrix
        Q = np.vstack((q))

        with pm.Model() as LR_model:

            # Priors for unknown model parameters
            k = pm.Uniform('k', lower=0.01, upper=params.kmax)

            # Priors for initial conditions and noise level
            sigma = pm.HalfNormal('sigma', sd=params.sdsigma)

            # Compute forward model
            forward = th_forward_lrmodel(k)

            # Compute likelihood
            Q_obs = pm.Lognormal('Q_obs', mu=pm.math.log(forward), sigma=sigma, observed=Q_sim)

            # Fix random seed
            np.random.seed(params.randomseed)

            # Initial points for each of the chains
            startsmc = [{'k':np.random.uniform(1e-3,params.kmax,1)} for _ in range(params.nchains)]

            # Sample posterior
            trace_LR = pm.sample(params.nsamples, progressbar=True, chains=params.nchains, start=startsmc, step=pm.SMC())

            return trace_LR

    def _print(self,trace):

        results = [pm.summary(trace, ['k'])]
        results = pd.concat(results)
        true_params.append(noise_sigma)
        results['True values'] = pd.Series(np.array(true_params), index=results.index)
        true_params.pop();

        print(results)

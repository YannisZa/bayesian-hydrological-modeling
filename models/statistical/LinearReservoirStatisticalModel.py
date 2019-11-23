from models.physical.ode_optimiser import ODEop,solveCached,Run
from models.physical.LinearReservoirModel import LinearReservoirModel as LRM
import numpy as np
import pymc3 as pm
import theano
from theano import *
import math
theano.config.exception_verbosity= 'high'
theano.config.floatX = 'float64'
print('Running on PyMC3 v{}'.format(pm.__version__))

n_states = 1
n_odeparams = 1
n_ivs = 0

class LinearReservoirStatisticalModel(object):

    def __init__(self, net_rainfall_data):
        self._nr = net_rainfall_data


    def run(self,q,params):

        # Define the data matrix
        Q = np.stack([q]).T

        # Define times
        times = np.arange(0,len(q))

        # Instantiate LRM model
        lrm = LRM(n_states, n_odeparams, n_ivs, self._nr)

        # Instantiate Run object
        r = Run(lrm, times, 1, 1)

        # Now instantiate the theano custom ODE op
        my_ODEop = ODEop(r.state,r.numpy_vsp)

        with pm.Model() as lrs_model:

            # Priors for unknown model parameters
            k =  pm.Uniform('k', lower=0.01, upper=params.kmax)

            q0 = pm.Lognormal('q0', mu=math.log(params.muq), sd=math.log(params.sdq))

            sigma = pm.Lognormal('sigma', mu=math.log(params.musigma), sd=math.log(params.sdsigma), shape=1)

            # Forward model
            all_params = pm.math.stack([k,q0],axis=0)
            ode_sol = my_ODEop(all_params)
            forward = ode_sol.reshape(Y.shape)

            # Likelihood
            Q_obs = pm.Lognormal('Q_obs', mu=pm.math.log(forward), sd=sigma, observed=Q)

            # Using Metropolis Hastings Sampling
            step = pm.Metropolis()

            # Draw the specified number of samples
            discharge_trace = pm.sample(params.N_SAMPLES, step=step, tune=( max(params.N_SAMPLES - 500, int(params.N_SAMPLES/3)) ), init='adapt_diag')

            return trace

from models.physical.ode_optimiser import ODEop,solveCached,Run
from models.physical.LinearReservoirModel import LinearReservoirModel as LRM
import numpy as np
import pymc3 as pm
import theano
from theano import *
import theano.tensor as tt
import math
theano.config.exception_verbosity= 'high'
theano.config.floatX = 'float64'
print('Running on PyMC3 v{}'.format(pm.__version__))

n_states = 1
n_odeparams = 1
n_ivs = 1

class LinearReservoirStatisticalModel(object):

    def __init__(self, net_rainfall_data):
        self._nr = net_rainfall_data


    def run(self,q,params):

        # Define the data matrix
        Q = np.vstack((q))

        # print(Q.shape)

        # Define times
        times = np.arange(0,len(q))

        # Instantiate Linear Reservoir model
        lrm = LRM(n_states, n_odeparams, n_ivs, self._nr)

        # Instantiate Run object
        r = Run(lrm, times, n_states, n_ivs, n_odeparams)

        # Now instantiate the theano custom ODE op
        my_ODEop = ODEop(r.state,r.numpy_vsp)

        with pm.Model() as lrs_model:

            # Priors for unknown model parameters
            k = pm.Uniform('k', lower=0.01, upper=params.kmax)
            # k = pm.Lognormal('k', mu=params.kmax, sd=1)

            # Priors for initial conditions and noise level
            q0 = pm.Lognormal('q0', mu=params.muq, sd=params.sdq)
            sigma = pm.Lognormal('sigma', mu=params.musigma, sd=params.sdsigma, shape=1)

            # Forward model
            all_params = pm.math.stack([k,q0],axis=0)
            ode_sol = my_ODEop(all_params)
            forward = ode_sol.reshape(Q.shape)

            # Print ODE solution
            # log_forward = pm.math.log(forward)
            # log_forward_print = tt.printing.Print('log_forward')(log_forward.shape)
            # tt.printing.Print('sigma')(sigma.shape)
            # print(np.log(Q))
            # print(Q)

            # Likelihood
            Q_obs = pm.Lognormal('Q_obs', mu=pm.math.log(forward), sd=sigma, observed=Q)
            # Q_obs_print = tt.printing.Print('Q_obs')(Q_obs)

            # for RV in lrs_model.basic_RVs:
            #     print(RV.name, RV.logp(lrs_model.test_point))
            print(lrs_model.check_test_point())

            # # Using Metropolis Hastings Sampling
            # step = pm.Metropolis()

            # Draw the specified number of samples
            # advi+adapt_diag
            discharge_trace = pm.sample(draws=params.nsamples, chains=params.nchains, tune=params.ntune, init='jitter+adapt_diag')

        print(discharge_trace['diverging'].sum())

        return discharge_trace

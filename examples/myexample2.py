import theano
from theano import *
import theano.tensor as tt
from theano.compile.ops import as_op
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import pymc3 as pm
import argparse
import json
import pickle
theano.config.exception_verbosity='high'

parser = argparse.ArgumentParser(description='Simulate discharge data and generate posterior samples using the linear reservoir model.')
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'simulations/linear_reservoir_simulation.csv',
                    help="filename of input dataframe (must end with .csv)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'posterior_samples/linear_reservoir_samples.pickle',
                    help="filename of output dataframe (must end with .csv)")
parser.add_argument("-q", "--q0",nargs='?',type=float,default = 0.1,
                    help="Initial discharge value Q(0) to be used in discharge data simulation")
parser.add_argument("-k", "--k",nargs='?',type=float,default = 8.0,
                    help="k is constant reaction factor or response factor with unit T (must be positive)\
                    used in discharge data simulation")
parser.add_argument("-s", "--sigma",nargs='?',type=float,default = 0.02,
                    help="Standard deviation of noise added to discharge simulation")
parser.add_argument("-kmax", "--kmax",nargs='?',type=float,default = 10.0,
                    help="k is constant reaction factor or response factor with unit T (must be positive) \
                        k ~ Uniform(0.01,kmax)")
parser.add_argument("-ss", "--sdsigma",nargs='?',type=float,default = 2.0,
                    help="Standard deviation of HalfNormal noise to be added to discharge  \
                    s ~ HarlfNormal(sdsigma^2), sdsigma is the prior standard deviation")
parser.add_argument("-ns", "--nsamples",nargs='?',type=int,default = 1000,
                    help="Number of posterior samples generated using choice of sampling method ")
parser.add_argument("-nc", "--nchains",nargs='?',type=int,default = 100,
                    help="Number of chains in posterior samples generation")
parser.add_argument("-r", "--randomseed",nargs='?',type=int,default = 24,
                    help="Random seed to be fixed when generating posterior samples")
args = parser.parse_args()


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


'''  Import data '''

df = pd.read_csv('/Users/Yannis/code/fibe2-mini-project/data/output/simulations/linear_reservoir_simulation.csv')
rf = df['rainfall'].values.tolist()
et = df['evapotranspiration'].values.tolist()
# Compute
nr = [max(rft - ett,1e-5) for rft, ett in zip(rf, et)]
q = df['discharge_approx'].values.tolist()
# q = [0.01, 0.084788051,0.289827287,0.487426902,0.623592162,0.855202214,0.901709887,0.87936577,0.857067839,0.775516564,0.701725939,0.675138958,0.68101658,0.64644605,0.701305112,0.747128907,0.676039744,0.668502137,0.731464651,0.766588801]
# nr = [1.618666063,0.0001,4.405308823,0.394073731,3.392555321,2.733285785,0.0001,1.31186209,0.0001,0.0001,0.0001,0.83074128,0.646141131,0.0001,2.405660466,0.0001,0.0001,1.174002978,1.481146447,0.73244669]

times = range(0,len(q))
q0 = args.q0
n_states = 1
n_times = len(q)
true_params = [int(args.k)]
noise_sigma = args.sigma
kmax = args.kmax
sdsigma = args.sdsigma
nsamples = args.nsamples
nchains = args.nchains
randomseed = args.randomseed

print(json.dumps(vars(args),indent=2))

# print('q0',q0)
# print('true_params',true_params)
# print('noise_sigma',noise_sigma)
# print('kmax',kmax)
# print('sdsigma',sdsigma)
# print('nsamples',nsamples)
# print('nchains',nchains)
# print('randomseed',randomseed)

print()
print()

# Define the data matrix
Q = np.vstack((q))

# Interpolate net rainfall data
nr_int = interp1d(times, nr,fill_value="extrapolate",kind='slinear')

ode_model = LinearReservoirModel(times,nr_int,q0)
sim_data = ode_model.simulate(true_params)
np.random.seed(42)

Q_sim = sim_data + np.random.randn(n_times,n_states)*noise_sigma

@as_op(itypes=[tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(param1):
    params = [param1]

    th_states = ode_model.simulate(params)
    return th_states


with pm.Model() as LR_model:

    # Priors for unknown model parameters
    k = pm.Uniform('k', lower=0.01, upper=kmax)
    # k = pm.Lognormal('k', mu=params.kmax, sd=1)

    # Priors for initial conditions and noise level
    #q0 = pm.Lognormal('q0', mu=muq, sd=sdq)
    sigma = pm.HalfNormal('sigma', sd=sdsigma)

    forward = th_forward_model(k)

    Q_obs = pm.Lognormal('Q_obs', mu=pm.math.log(forward), sigma=sigma, observed=Q_sim)

    # Initial points for each of the chains
    np.random.seed(randomseed)
    startsmc = [{'k':np.random.uniform(0.01,kmax,1)} for _ in range(nchains)]

    trace_LR = pm.sample(nsamples, progressbar=True, chains=nchains, start=startsmc, step=pm.SMC())



results=[pm.summary(trace_LR, ['k']),pm.summary(trace_LR, ['sigma'])]
results=pd.concat(results)
true_params.append(noise_sigma)
results['True values'] = pd.Series(np.array(true_params), index=results.index)
true_params.pop();
print(results)

with open(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',output_filename), 'wb') as buff:
    pickle.dump(sample_trace, buff)


print('Posterior computed and saved to...')
print(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',output_filename))

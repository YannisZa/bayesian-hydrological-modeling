import sys
import os
sys.path.append(os.getcwd())
import theano
from theano import *
import theano.tensor as tt
from theano.compile.ops import as_op
from models.LinearReservoirModel import LinearReservoirModel as LRM
import pandas as pd
import numpy as np
import pymc3 as pm
import argparse
import pickle
import json
import math

def print_model_specification(args):
    print(f'k ~ Uniform(0.01,{args.kmax})')
    print(f'sigma ~ HalfNormal({args.sdsigma})')


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

print(json.dumps(vars(args),indent=2))
print_model_specification(args)
print()

'''  Import data '''

df = pd.read_csv(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.input_filename))
rf = df['rainfall'].values.tolist()
et = df['evapotranspiration'].values.tolist()
# Compute
nr = [max(rft - ett,1e-5) for rft, ett in zip(rf, et)]
q = df['discharge_approx'].values.tolist()

''' Compute posterior samples '''

# Instantiate linear reservoir statistical model
lrm = LRM(nr,args)

@as_op(itypes=[tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(param1):
    parameter_list = [param1]

    th_states = lrm.simulate(parameter_list)
    return th_states

# Simulate Q data
sim_data = lrm.simulate([args.k])
# Fix random seed
np.random.seed(42)
# Add noise to Q
Q_sim = sim_data + np.random.randn(lrm._n_times,lrm._n_states)*args.sigma

# Define the data matrix
Q = np.vstack((q))

# Define true parameters
true_params = [args.k]

with pm.Model() as LR_model:

    # Priors for unknown model parameters
    k = pm.Uniform('k', lower=0.01, upper=args.kmax)

    # Priors for initial conditions and noise level
    sigma = pm.HalfNormal('sigma', sd=args.sdsigma)

    # Compute forward model
    forward = th_forward_model(k)

    # Compute likelihood
    Q_obs = pm.Lognormal('Q_obs', mu=pm.math.log(forward), sigma=sigma, observed=Q_sim)

    # Fix random seed
    np.random.seed(args.randomseed)

    # Initial points for each of the chains
    startsmc = [{'k':np.random.uniform(0.01,args.kmax,1)} for _ in range(args.nchains)]

    # Sample posterior
    trace_LR = pm.sample(args.nsamples, progressbar=True, chains=args.nchains, start=startsmc, step=pm.SMC())

    # Return results, too
    results = [pm.summary(trace_LR, ['k']),pm.summary(trace_LR, ['sigma'])]
    results = pd.concat(results)
    true_params.append(args.sigma)
    results['True values'] = pd.Series(np.array(true_params), index=results.index)
    true_params.pop();
    print(results)

# Save trace as pickle
with open(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename), 'wb') as buff:
    pickle.dump(sample_trace, buff)

# Save results as csv
results.to_csv(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename.replace('.pickle','.csv')), index = False)

print('Posterior computed and saved to...')
print(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename.replace('.pickle','')))

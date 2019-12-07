import os
import sys
sys.path.append(os.getcwd())
import theano
from theano import *
import theano.tensor as tt
from theano.compile.ops import as_op
from models.LinearReservoirModel import LinearReservoirModel as LRM
from argparse import Namespace
from tqdm import tqdm
import pandas as pd
import numpy as np
import pymc3 as pm
import argparse
import pickle
import json
import math
import resource
import traceback

# Run with THEANO_FLAGS=mode=FAST_RUN

# Set maximum number of open files
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

def print_model_specification(args):
    print(f'k ~ Uniform(0.01,{args.kmax})')
    print(f'sigma ~ Gamma({args.alpha},{args.beta})')
    print('Q(t) ~ Normal(Q(q0,t),sigma)')
    print()


parser = argparse.ArgumentParser(description='Simulate discharge data and generate posterior samples using the linear reservoir model.')
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'simulations/linear_reservoir_simulation.csv',
                    help="filename of input dataframe (must end with .csv) (default: %(default)s)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'posterior_samples/linear_reservoir_samples.pickle',
                    help="filename of output dataframe (must end with .csv) (default: %(default)s)")
parser.add_argument("-kmax", "--kmax",nargs='?',type=float,default = 10.0,
                    help="k is constant reaction factor or response factor with unit T (must be positive) \
                        k ~ Uniform(0.01,kmax) (default: %(default)s)")
parser.add_argument("-a", "--alpha",nargs='?',type=float,default = 2.0,
                    help="Hyperparameter for Gaussian noise N(0,s) added to discharge  \
                    sigma ~ Gamma(alpha,beta), alpha is the shape factor (default: %(default)s)")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 40.0,
                    help="Hyperparameter for Gaussian noise N(0,s) added to discharge  \
                    sigma ~ Gamma(alpha,beta), beta is the rate factor (default: %(default)s)")
parser.add_argument("-ns", "--nsamples",nargs='?',type=int,default = 1000,
                    help="Number of posterior samples generated using choice of sampling method (default: %(default)s)")
parser.add_argument("-nc", "--nchains",nargs='?',type=int,default = 100,
                    help="Number of chains in posterior samples generation (default: %(default)s)")
parser.add_argument("-r", "--randomseed",nargs='?',type=int,default = 24,
                    help="Random seed to be fixed when generating posterior samples (default: %(default)s)")
args = parser.parse_args()
params = vars(args)

# Get current working directory and project root directory
cwd = os.getcwd()
rd = os.path.join(cwd.split('fibe2-mini-project/', 1)[0])
if not rd.endswith('fibe2-mini-project'):
    rd = os.path.join(cwd.split('fibe2-mini-project/', 1)[0],'fibe2-mini-project')

# Export model priors
with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle','_priors.json')), 'w') as f:
    json.dump(params, f)

print(json.dumps(params,indent=2))
print_model_specification(args)
print()


'''  Import simulated data '''

# Import simulated data from all three models
model0data = pd.read_csv(os.path.join(rd,'data','output','simulations','linear_reservoir_simulation.csv'))
model1data = pd.read_csv(os.path.join(rd,'data','output','simulations','nonlinear_reservoir_simulation.csv'))
model2data = pd.read_csv(os.path.join(rd,'data','output','simulations','hymod_simulation.csv'))

# Store net net_rainfall
nr = model0data['net_rainfall'].values.tolist()
n = len(nr)

# Import Q(0)
with open(os.path.join(rd,'data','output',args.input_filename.replace('.csv','_true_parameters.json')), 'r') as f:
    true_params = json.load(f)
true_args = Namespace(**true_params)

# Store simulated discharges from three models
model0q = model0data['discharge'].values.reshape(n,1)
model1q = model1data['discharge'].values.reshape(n,1)
model2q = model2data['discharge'].values.reshape(n,1)

# Add model dischaged to dictionary
model_discharges = {'LRM':model0q,'NLRM':model1q,'HYMOD':model2q}

''' Compute posterior samples '''

# Instantiate linear reservoir statistical model
lrm = LRM(nr,true_args)

@as_op(itypes=[tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(param1):
    parameter_list = [param1]

    th_states = lrm.simulate(parameter_list,true_args.fatconv)
    return th_states

# Initialise dataframe to store parameter posteriors
keys = ['current_model','true_model','parameter','log_marginal_likelihood','mean', 'sd', 'mc_error', 'hpd_2.5', 'hpd_97.5']
results = pd.DataFrame(columns=keys)

# Initialise empty model and trace dictionaries
models = {}
traces = {}

# Loop over simulated datasets and compute marginal
for mi in tqdm(model_discharges.keys()):

    print(f'LRM posterior sample generation using {mi} data')

    with pm.Model() as LR_model:

        # Priors for unknown model parameters
        k = pm.Uniform('k', lower=0.01, upper=args.kmax)

        # Priors for initial conditions and noise level
        sigma = pm.Gamma('sigma',alpha=args.alpha,beta=args.beta)

        # Compute forward model
        # try:
        forward = th_forward_model(k)
        # except AssertionError:
        #     _, _, tb = sys.exc_info()
        #     traceback.print_tb(tb) # Fixed format
        #     tb_info = traceback.extract_tb(tb)
        #     filename, line, func, text = tb_info[-1]
        #
        #     print('An error occurred on line {} in statement {}'.format(line, text))
        #     exit(1)

        # Compute likelihood
        Q_obs = pm.Normal('Q_obs', mu=forward, sigma=sigma, observed=model_discharges[mi])

        # Fix random seed
        np.random.seed(args.randomseed)

        # Initial points for each of the chains
        startsmc = [{'k':np.random.uniform(0.01,args.kmax,1)} for _ in range(args.nchains)]

        # Sample posterior
        trace_LR = pm.sample(args.nsamples, progressbar=True, chains=args.nchains, start=startsmc, step=pm.SMC())

        # Compute negative log marginal likelihood
        log_ml = -np.log(LR_model.marginal_likelihood)

        # Append to results
        vals = np.append(np.array(['LRM',mi,'k',log_ml]),pm.summary(trace_LR, ['k']).values[0])
        results = results.append(dict(zip(keys, vals)),ignore_index=True)
        vals = np.append(np.array(['LRM',mi,'sigma',log_ml]),pm.summary(trace_LR, ['sigma']).values[0])
        results = results.append(dict(zip(keys, vals)),ignore_index=True)

        # Append to models and traces
        # models[mi] = LR_model
        # traces[mi] = trace_LR

        # Save model as pickle
        with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle',f'_{mi}data_model.pickle')), 'wb') as buff1:
            pickle.dump(LR_model, buff1)

        # Save trace as pickle
        with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle',f'_{mi}data_trace.pickle')), 'wb') as buff2:
            pickle.dump(trace_LR, buff2)

        # Save results as csv
        results.to_csv(os.path.join(rd,'data','output',args.output_filename.replace('.pickle','_summary.csv')), index = False)

        print('Results so far...')
        print(results)
        print()


# Set results df index
results = results.set_index(['current_model','true_model','parameter'])

# Save results as csv
results.to_csv(os.path.join(rd,'data','output',args.output_filename.replace('.pickle','_summary.csv')), index = False)

# # Export models
# for mi in tqdm(models.keys()):
#     # Save model as pickle
#     with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle',f'_{mi}data_model.pickle')), 'wb') as buff1:
#         pickle.dump(models[mi], buff1)
#
# # Export traces
# for mi in tqdm(traces.keys()):
#     # Save trace as pickle
#     with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle',f'_{mi}data_trace.pickle')), 'wb') as buff2:
#         pickle.dump(traces[mi], buff2)


print('Posterior computed and saved to...')
print(os.path.join(rd,'data','output',args.output_filename.replace('.pickle','')))
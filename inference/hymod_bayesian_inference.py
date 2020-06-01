import os
import sys
sys.path.append(os.getcwd())
import theano
from theano import *
import theano.tensor as tt
from theano.compile.ops import as_op
from models.HymodModel import HymodModel as HYMOD
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

def print_model_specification(args):
    print(f'cmax ~ Uniform(1.0,{args.c_max})')
    print(f'k_slow ~ Uniform(0.01,{args.kslow_max})')
    print(f'k_fast ~ Uniform(0.01,{args.kfast_max})')
    print(f'beta_k ~ Gamma({args.betak_alpha},{args.betak_beta})')
    print(f'alfa ~ Beta({args.alfa_alpha},{args.alfa_beta})')
    print(f'sigma ~ Gamma({args.sigma_alpha},{args.sigma_beta})')
    print('Q(t) ~ Normal(Q(q0,t),sigma)')
    print()


parser = argparse.ArgumentParser(description='Simulate discharge data and generate posterior samples using the HYMOD model.')
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'simulations/hymod_simulation_monthly.csv',
                    help="filename of input dataframe (must end with .csv) (default: %(default)s)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'posterior_samples/hymod_samples_monthly.pickle',
                    help="filename of output dataframe (must end with .csv) (default: %(default)s)")

parser.add_argument("-cmax", "--c_max",nargs='?',type=float,default = 400.0, #500.0§
                    help="maximum soil water storage in length units \
                    c_{max} ~ Uniform(0.01,cmax) -- default: %(default)s")
parser.add_argument("-ba", "--betak_alpha",nargs='?',type=float,default = 2.0,
                    help="shape factor of the main soil-water storage tank that represents the degree of spatial variability of the soil-moisture capacity within the catchment \
                    beta_{k} ~ Gamma(betak_alpha,betak_beta) betak_alpha is mean -- default: %(default)s")
parser.add_argument("-bb", "--betak_beta",nargs='?',type=float,default = 6.0,
                    help="shape factor of the main soil-water storage tank that represents the degree of spatial variability of the soil-moisture capacity within the catchment \
                    beta_{k} ~ Gamma(betak_alpha,betak_beta) betak_beta is sigma -- default: %(default)s")
parser.add_argument("-ala", "--alfa_alpha",nargs='?',type=float,default = 2.0,
                    help="factor distributing flow between two series of reservoirs \
                    alfa ~ Beta(alfa_alpha,alfa_beta) alfa_alpha is shape factor -- default: %(default)s")
parser.add_argument("-alb", "--alfa_beta",nargs='?',type=float,default = 3.0,
                    help="factor distributing flow between two series of reservoirs \
                    alfa ~ Beta(alfa_alpha,alfa_beta) alfa_beta is scale factor -- default: %(default)s")
parser.add_argument("-kf", "--kfast_max",nargs='?',type=float,default = 2.0,
                    help="fast runoff: constant reaction factor or response factor with unit T (must be positive) (default: %(default)s) \
                    K_{fast} ~ Uniform(0.01,kfastmax) -- default: %(default)s")
parser.add_argument("-ks", "--kslow_max",nargs='?',type=float,default = 2.0,
                    help="slow runoff: constant reaction factor or response factor with unit T (must be positive) (default: %(default)s) \
                    K_{slow} ~ Uniform(0.01,kslowmax) -- default: %(default)s")
parser.add_argument("-nr", "--nreservoirs",nargs='?',type=int,default = 3,
                    help="number of linear reservoirs for fast flow to be cascaded (default: %(default)s)")
parser.add_argument("-sa", "--sigma_alpha",nargs='?',type=float,default = 2.0,
                    help="Hyperparameter for Gaussian noise N(0,s) added to discharge  \
                    sigma ~ Gamma(sigma_alpha,sigma_beta), alpha is the shape factor -- default: %(default)s")
parser.add_argument("-sb", "--sigma_beta",nargs='?',type=float,default = 4.0,
                    help="Hyperparameter for Gaussian noise N(0,s) added to discharge  \
                    sigma ~ Gamma(sigma_alpha,sigma_beta), beta is the rate factor -- default: %(default)s")
parser.add_argument("-ns", "--nsamples",nargs='?',type=int,default = 2000,
                    help="Number of posterior samples generated using choice of sampling method (default: %(default)s)")
parser.add_argument("-nc", "--nchains",nargs='?',type=int,default = 200,
                    help="Number of chains in posterior samples generation (default: %(default)s)")
parser.add_argument("-r", "--randomseed",nargs='?',type=int,default = 24,
                    help="Random seed to be fixed when generating posterior samples (default: %(default)s)")
args = parser.parse_args()
params = vars(args)

# Get current working directory and project root directory
cwd = os.getcwd()
rd = os.path.join(cwd.split('bayesian-hydrological-modeling/', 1)[0])
if not rd.endswith('bayesian-hydrological-modeling'):
    rd = os.path.join(cwd.split('bayesian-hydrological-modeling/', 1)[0],'bayesian-hydrological-modeling')

# Export model priors
with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle','_priors.json')), 'w') as f:
    json.dump(params, f)

print(json.dumps(params,indent=2))
print_model_specification(args)
print()

# sys.exit()

'''  Import simulated data '''

# Import simulated data from all three models
model0data = pd.read_csv(os.path.join(rd,'data','output','simulations','linear_reservoir_simulation_monthly.csv'))
model1data = pd.read_csv(os.path.join(rd,'data','output','simulations','nonlinear_reservoir_simulation_monthly.csv'))
model2data = pd.read_csv(os.path.join(rd,'data','output','simulations','hymod_simulation_monthly.csv'))

# Store rainfall and evapotranspiration
rn = model0data['net_rainfall'].values.tolist()
et = model0data['evapotranspiration'].values.tolist()
n = len(rn)

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
hymod = HYMOD(rn,et,true_args)

@as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(param1,param2,param3,param4,param5):
    parameter_list = [param1,param2,param3,param4,param5]

    th_states = hymod.simulate(parameter_list,true_args)
    return th_states

# Initialise dataframe to store parameter posteriors
keys = ['current_model','true_model','parameter','marginal_likelihood','mean', 'sd', 'mc_error', 'hpd_2.5', 'hpd_97.5']
results = pd.DataFrame(columns=keys)

# Loop over simulated datasets and compute marginal
for mi in tqdm(model_discharges.keys()):

    print(f'HYMOD posterior sample generation using {mi} data')

    with pm.Model() as HYMOD_model:

        # Priors for unknown model parameters
        cmax = pm.Uniform('cmax',lower=1.0,upper=args.c_max,transform=None)
        kfast = pm.Uniform('kfast',lower=0.01,upper=args.kfast_max,transform=None)
        kslow = pm.Uniform('kslow',lower=0.01,upper=args.kslow_max,transform=None)
        betak = pm.Gamma('betak', alpha=args.betak_alpha,beta=args.betak_beta,transform=None)
        alfa = pm.Beta('alfa',alpha=args.alfa_alpha,beta=args.alfa_beta,transform=None)

        # Priors for initial conditions and noise level
        sigma = pm.Gamma('sigma',alpha=args.sigma_alpha,beta=args.sigma_beta)

        # Compute forward model
        forward = th_forward_model(cmax,betak,alfa,kfast,kslow)

        # Compute likelihood
        Q_obs = pm.Normal('Q_obs', mu=forward, sigma=sigma, observed=model_discharges[mi])


        # Fix random seed
        np.random.seed(args.randomseed)

        # Initial points for each of the chains
        startsmc = [{'cmax':np.random.uniform(0.01,args.c_max,1),
                    'kfast':np.random.uniform(0.01,args.kfast_max,1),
                    'kslow':np.random.uniform(0.01,args.kslow_max,1),
                    'betak':np.random.gamma(shape=args.betak_alpha,scale=1/args.betak_beta),
                    'alfa':np.random.beta(a=args.alfa_alpha,b=args.alfa_beta)}
                    for _ in range(args.nchains)]

        # Sample posterior
        # trace_HYMOD = pm.sample(args.nsamples, progressbar=True, start=startsmc, step=pm.SMC(),random_seed=args.randomseed)
        trace_HYMOD = pm.sample(args.nsamples, progressbar=True, step=pm.SMC(),random_seed=args.randomseed)

        # Compute negative marginal likelihood
        ml = HYMOD_model.marginal_likelihood #-np.log(HYMOD_model.marginal_likelihood)
        print('Marginal Likelihood:',ml)

        # Append to results
        for key in ['cmax','kfast','kslow','betak','alfa','sigma']:
            vals = np.append(np.array(['HYMOD',mi,key,ml]),pm.summary(trace_HYMOD, [key]).values[0])
            results = results.append(dict(zip(keys, vals)),ignore_index=True)

        # Save model as pickle
        with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle',f'_{mi}data_model.pickle')), 'wb') as buff1:
            pickle.dump(HYMOD_model, buff1)

        # Save trace as pickle
        with open(os.path.join(rd,'data','output',args.output_filename.replace('.pickle',f'_{mi}data_trace.pickle')), 'wb') as buff2:
            pickle.dump(trace_HYMOD, buff2)

        # Save results as csv
        results.to_csv(os.path.join(rd,'data','output',args.output_filename.replace('.pickle',f'_{mi}summary.csv')), index = False)

        print('Results so far...')
        print(results.head(results.shape[0]))
        print()


# Set results df index
results = results.set_index(['current_model','true_model','parameter'])

# Save results as csv
results.to_csv(os.path.join(rd,'data','output',args.output_filename.replace('.pickle','_summary.csv')), index = False)

print('Posterior computed and saved to...')
print(os.path.join(rd,'data','output',args.output_filename.replace('.pickle','')))

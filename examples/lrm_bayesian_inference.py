import sys
import os
sys.path.append(os.getcwd())
from models.statistical.LinearReservoirStatisticalModel import LinearReservoirStatisticalModel as LRSM
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import pandas as pd
import numpy as np
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
parser.add_argument("-s", "--sigma",nargs='?',type=float,default = 0.1,
                    help="Standard deviation of noise added to discharge simulation")
parser.add_argument("-kmax", "--kmax",nargs='?',type=float,default = 10.0,
                    help="k is constant reaction factor or response factor with unit T (must be positive) \
                        k ~ Uniform(0.01,kmax)")
parser.add_argument("-ss", "--sdsigma",nargs='?',type=float,default = 2.0,
                    help="Standard deviation of HalfNormal noise to be added to discharge  \
                    s ~ HarlfNormal(sdsigma^2), sdsigma is the prior standard deviation")
parser.add_argument("-ns", "--nsamples",nargs='?',type=int,default = 1000,
                    help="Number of posterior samples generated using choice of sampling method ")
parser.add_argument("-nc", "--nchains",nargs='?',type=int,default = 500,
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
q = df['discharge_approx'].values

''' Compute posterior samples '''

# Instantiate linear reservoir statistical model
lrsm = LRSM(nr,args)

# Simulate data
Qsim = lrsm._simulate_q(args)

# Compute posterior samples
sample_trace = lrsm._sample(q,Qsim,args)

# Print sample trace results
lrsm._print(sample_trace)

with open(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename), 'wb') as buff:
    pickle.dump(sample_trace, buff)


print('Posterior computed and saved to...')
print(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename))

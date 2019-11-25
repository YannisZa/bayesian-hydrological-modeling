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
    print(f'q0 ~ LogNormal({args.muq},{args.sdq})')
    print(f'sigma ~ LogNormal({args.musigma},{args.sdsigma})')


parser = argparse.ArgumentParser(description='Simulate discharge data using the linear reservoir model.')
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'simulations/linear_reservoir_simulation.csv',
                    help="filename of input dataframe (must end with .csv)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'posterior_samples/linear_reservoir_samples.pickle',
                    help="filename of output dataframe (must end with .csv)")
parser.add_argument("-kmax", "--kmax",nargs='?',type=float,default = 10.0,
                    help="k is constant reaction factor or response factor with unit T (must be positive) \
                        k ~ Uniform(0.01,kmax)")
parser.add_argument("-mq", "--muq",nargs='?',type=float,default = math.log(0.1),
                    help="q0 is the value of discharge at time 0 (must be positive) \
                    q0 ~ LogNormal(log(muq),log(sdq^2)), muq is the prior mean")
parser.add_argument("-sq", "--sdq",nargs='?',type=float,default = 2,
                    help="q0 is the value of discharge at time 0 (must be positive) \
                    q0 ~ LogNormal(log(muq),log(sdq^2)), sdq is the prior standard deviation")
parser.add_argument("-ms", "--musigma",nargs='?',type=float,default = -1.0,
                    help="Standard deviation of Log-Gaussian noise to be added to discharge  \
                    s ~ LogNormal(log(musigma), log(sdsigma)^2), musigma is the prior mean")
parser.add_argument("-ss", "--sdsigma",nargs='?',type=float,default = 2.0,
                    help="Standard deviation of Log-Gaussian noise to be added to discharge  \
                    s ~ LogNormal(log(musigma), log(sdsigma)^2), sdsigma is the prior standard deviation")
parser.add_argument("-ns", "--nsamples",nargs='?',type=int,default = 1500,
                    help="Number of posterior samples generated using choice of sampling method ")
parser.add_argument("-nt", "--ntune",nargs='?',type=int,default = 1000,
                    help="Number of samples used for tuning in posterior sample generation")
parser.add_argument("-nc", "--nchains",nargs='?',type=int,default = 1,
                    help="Number of chains in posterior samples generation")
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
lrsm = LRSM(nr)
# Compute posterior samples
sample_trace = lrsm.run(q,args)

with open(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename), 'wb') as buff:
    pickle.dump(sample_trace, buff)

print('Posterior computed and saved to...')
print(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename))

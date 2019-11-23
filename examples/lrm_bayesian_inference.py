import sys
import os
sys.path.append(os.getcwd())
from models.statistical.LinearReservoirStatisticalModel import LinearReservoirStatisticalModel as LRSM
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import pandas as pd
import numpy as np
import argparse
import json
import math

def print_model_specification(args):

    print(f'k ~ Uniform(0.01,{args.kmax})')
    print(f'q0 ~ LogNormal(log({args.muq}),log({args.sdq}))')
    print(f'sigma ~ LogNormal(log({args.musigma}),log({args.sdsigma}))')


parser = argparse.ArgumentParser(description='Simulate discharge data using the linear reservoir model.')
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'linear_reservoir_simulation.csv',
                    help="filename of input dataframe (must end with .csv)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'posterior_samples/linear_reservoir_samples.csv',
                    help="filename of output dataframe (must end with .csv)")
parser.add_argument("-kmax", "--kmax",nargs='?',type=float,default = 10.0,
                    help="k is constant reaction factor or response factor with unit T (must be positive) \
                        k ~ Uniform(0,kmax)")
parser.add_argument("-mq", "--muq",nargs='?',type=float,default = 1.0,
                    help="q0 is the value of discharge at time 0 (must be positive) \
                    q0 ~ LogNormal(log(muq),log(sdq^2)), muq is the prior mean")
parser.add_argument("-sq", "--sdq",nargs='?',type=float,default = 0.1,
                    help="q0 is the value of discharge at time 0 (must be positive) \
                    q0 ~ LogNormal(log(muq),log(sdq^2)), sdq is the prior standard deviation")
parser.add_argument("-ms", "--musigma",nargs='?',type=float,default = 1,
                    help="Standard deviation of white Gaussian noise N(0,s^2) to be added to discharge  \
                    s ~ LogNormal(log(musigma), log(sdsigma)), musigma is the prior mean")
parser.add_argument("-ss", "--sdsigma",nargs='?',type=float,default = 0.1,
                    help="Standard deviation of white Gaussian noise N(0,s^2) to be added to discharge  \
                    s ~ LogNormal(log(musigma), log(sdsigma)), sdsigma is the prior standard deviation")
parser.add_argument("-n", "--N_SAMPLES",nargs='?',type=int,default = 1500,
                    help="Number of posterior samples generated using choice of sampling method ")
args = parser.parse_args()

print(json.dumps(vars(args),indent=2))
print_model_specification(args)

'''  Import data '''

df = pd.read_csv(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.input_filename))
rf = df['rainfall'].values.tolist()
et = df['evapotranspiration'].values.tolist()
# Compute
nr = [rft - ett for rft, ett in zip(rf, et)]
q = df['discharge_approx'].values.tolist()

''' Compute posterior samples '''

# Instantiate linear reservoir statistical model
lrsm = LRSM(nr)
# Compute posterior samples
sample_trace = lrsm.run(q,args)

print(sample_trace)

# sample_df = pd.DataFrame([[range(1,(len(samples)+1)), samples]],columns = ['iteration','sample'])
# sample_df.to_csv(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename),index = False)

print('Posterior computed')

import pandas as pd
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from sympy.solvers.ode import dsolve
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse
import random
import json
import math
import os
#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Simulate discharge data using the non-linear reservoir model.')
parser.add_argument("--simulate",dest='simulate', action='store_true',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to true")
parser.add_argument("--no-simulate",dest='simulate', action='store_false',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to false")
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'synthetic/rainfall_evapotranspiration_syn.csv',
                    help="filename of input dataframe (must end with .csv)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'nonlinear_reservoir_simulation.csv',
                    help="filename of output dataframe (must end with .csv)")
parser.add_argument("-k", "--k",nargs='?',type=float,default = 10.0,
                    help="constant reaction factor or response factor with unit T (must be positive)")
parser.add_argument("-m", "--m",nargs='?',type=float,default = 0.3,
                    help="store exponent (must be positive)")
parser.add_argument("-q", "--q0",nargs='?',type=float,default = 0.01,
                    help="value of discharge (q) at time 0 (must be positive)")
parser.add_argument("-tdelta", "--tdelta",nargs='?',type=int,default = 3600,
                    help="timestep to normalise time by to make the time units seconds")
parser.add_argument("-s", "--sigma_noise",nargs='?',type=float,default = 0.05,
                    help="Standard deviation of white Gaussian noise N(0,s^2) to be added to discharge ")
args = parser.parse_args()

# Flag whether to simulate or use existing data
simulate = args.simulate

# Fixed parameters
m = args.m # store exponent :: - :: m > 0
k = args.k # constant reaction factor or response factor :: Time :: k > 0

q0 = args.q0 # Let Q(0) = 0 - Discharge at time zero is zero
tdelta = args.tdelta # num of seconds in day/hour etc. depending on data frequency

# Gaussian error noise
sigma_noise = args.sigma_noise

print(json.dumps(vars(args),indent=2))

''' Non-linear reservoir model

dQ(t)/dt = a(R(t) - Q(t)) * (Q(t))^b
Q(t) = (S(t))^m/K

where
    a = m*(1/K)^(1/m) :: a > 0
    b = (m-1)/m :: -infty < b < 1


Net rainfal is equal to precipitation minus potential evapotranspiration
R(t) = P(t) - E_pt(t)

'''

# Compute derived parameters
a = m*(1./k)**(1./m)
b = (m-1)/m

# Raise ValueError if m is negative
if m <= 0:
    print(f'Number of reservoirs {m} has to be positive.')
    raise ValueError('Please choose another value for m.')

if simulate:
    print('Simulating input data')
    # Simulate rainfall and evapotranspiration data
    rf = np.concatenate((np.random.normal(3, 1, 100) , np.random.normal(3, 0.5, 100)))
    # Some days have no rainfall
    zero_rf_days = random.sample(range(0,len(rf)),30)
    rf[zero_rf_days] = 0

    et = np.concatenate((np.random.normal(2.5, 1, 100) , np.random.normal(4, 0.5, 100)))
    # Some days have no evapotranspiration
    zero_et_days = random.sample(range(0,len(et)),30)
    et[zero_et_days] = 0

    # Compute
    nr = [rft - ett for rft, ett in zip(rf, et)]

else:
    print('Reading input data')
    df = pd.read_csv(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/input/',args.input_filename))
    rf = df['rainfall'].values.tolist()
    et = df['evapotranspiration'].values.tolist()
    # Compute
    nr = [rft - ett for rft, ett in zip(rf, et)]

n = len(nr)

# time points
time = range(0,n)
#t = np.linspace(start=0,stop=100000,num=n_samples)#range(0,n_samples)

# Interpolate net_rainfall
nr_int = interp1d(time, nr,fill_value="extrapolate",kind='slinear')

# Define model
def nonlinear_reservoir_model(q,time,nrint):
    return ( a * ( nrint(time) - max(q,0.01) ) * (max(q,0.01)**b) )


# Solve ODE
q = odeint(nonlinear_reservoir_model,q0,time,args=(nr_int,))
# Flatten q
q_flat = [max(item,0.0) for sublist in q for item in sublist]

# Generate random errors
error = np.random.normal(0,sigma_noise,len(q_flat))
# Add Gaussian noise to discharge
q_flat = [max(q+e,0) for q,e in zip(q_flat,error)]


''' Export data to file '''

filename = os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename)
df = pd.DataFrame(list(zip(time,rf,et,nr,q_flat)), columns =['time', 'rainfall','evapotranspiration','net_rainfall', 'discharge_approx'])
df.to_csv(filename,index=False)

print('Done!...')

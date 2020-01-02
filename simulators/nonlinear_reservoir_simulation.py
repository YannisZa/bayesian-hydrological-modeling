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
import sys
import os

''' Non-linear reservoir model

dQ(t)/dt = a(R(t) - Q(t)) * (Q(t))^b
Q(t) = (S(t))^m/K

where
    a = m*(1/K)^(1/m) :: a > 0
    b = (m-1)/m :: -infty < b < 1


Net rainfal is equal to precipitation minus potential evapotranspiration
R(t) = P(t) - E_pt(t)

'''

parser = argparse.ArgumentParser(description='Simulate discharge data using the non-linear reservoir model.')
parser.add_argument("--simulate",dest='simulate', action='store_true',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to true")
parser.add_argument("--no-simulate",dest='simulate', action='store_false',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to false")
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'raw/road_data.csv',
                    help="filename of input dataframe (must end with .csv) (default: %(default)s)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'simulations/nonlinear_reservoir_simulation.csv',
                    help="filename of output dataframe (must end with .csv) (default: %(default)s)")
parser.add_argument("-k", "--k",nargs='?',type=float,default = 1.1,
                    help="constant reaction factor or response factor with unit T (must be positive) (default: %(default)s)")
parser.add_argument("-m", "--m",nargs='?',type=float,default = 1.5,
                    help="store exponent (must be positive)")
parser.add_argument("-q", "--q0",nargs='?',type=float,default = 0.01,
                    help="value of discharge (q) at time 0 (must be positive)")
parser.add_argument("-s", "--sigma",nargs='?',type=float,default = 0.5,
                    help="Standard deviation of white Gaussian noise N(0,s^2) to be added to discharge (default: %(default)s)")
parser.add_argument("-a", "--catchment_area",nargs='?',type=float,default = 15966.0,
                    help="Area of catchment (in m^2) to be multiplied with discharge (default: %(default)s)")
parser.add_argument("-t", "--tdelta",nargs='?',type=int,default = 1,
                    help="timestep to normalise time by to make the time units seconds (default: %(default)s)")
parser.add_argument("-r", "--randomseed",nargs='?',type=int,default = 22,
                    help="fixed random seed for generating noise")
args = parser.parse_args()
params = vars(args)

minvalue = 0.0001

# Get current working directory and project root directory
cwd = os.getcwd()
rd = os.path.join(cwd.split('fibe2-mini-project/', 1)[0])
if not rd.endswith('fibe2-mini-project'):
    rd = os.path.join(cwd.split('fibe2-mini-project/', 1)[0],'fibe2-mini-project')

# Set conversion factor
args.fatconv = 1. / 1000.0 / args.tdelta * args.catchment_area
print('fatconv',args.fatconv)

# Export model parameters
with open(os.path.join(rd,'data','output',args.output_filename.replace('.csv','_true_parameters.json')), 'w') as f:
    json.dump(params, f)


print(json.dumps(params,indent=2))


''' Generate or read discharge data '''

# Compute derived parameters
a = args.m*(1./args.k)**(1./args.m)
b = (args.m-1)/args.m

print('a',a)
print('b',b)

# Raise ValueError if m is negative
if args.m <= 0:
    print(f'Number of reservoirs {args.m} has to be positive.')
    raise ValueError('Please choose another value for m.')

if args.simulate:
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
    nr = [max(rft - ett,minvalue) for rft, ett in zip(rf, et)]

else:
    print('Reading input data')
    df = pd.read_csv(os.path.join(rd,'data','input',args.input_filename))
    rf = df['rainfall'].values.tolist()
    et = df['evapotranspiration'].values.tolist()
    # Compute
    nr = [max(rft - ett,minvalue) for rft, ett in zip(rf, et)]

# Number of timesteps
n = len(nr)

# time points
time = range(0,n)

# Interpolate net_rainfall
nr_int = interp1d(time, nr,fill_value="extrapolate",kind='slinear')

# Define model
def nonlinear_reservoir_model(q,t,nrint):
    return ( a * ( max(minvalue,nrint(t)) - q) * (q**b) )


# Solve ODE
q = odeint(nonlinear_reservoir_model,args.q0,time,args=(nr_int,))
# Flatten q
q_flat = [item * args.fatconv for sublist in q for item in sublist]
q_flat = [q_flat[i] for i in range(0,n)]

# Fix random seed
np.random.seed(args.randomseed)
# Add Gaussian noise to discharge
Q_sim = np.asarray(q_flat).reshape(n,1) + np.random.randn(n,1)*args.sigma
Q_sim = [max(minvalue,qsim[0]) for qsim in Q_sim]

''' Export data to file '''
df = pd.DataFrame(list(zip(time,rf,et,nr,Q_sim)), columns =['time', 'rainfall','evapotranspiration','net_rainfall', 'discharge'])
print(df.head(10))

df.to_csv(os.path.join(rd,'data','output',args.output_filename),index=False)

print('Done!...')

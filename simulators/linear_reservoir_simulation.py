import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse
import random
import json
import math
import sys
import os

''' Linear reservoir model

Q(t) = R(t) - KdQ(t)/dt
Q(t) = S(t)/K

where R(t) = P(t) - E_pt(t)

States
Q(t) :: discharge at time t (in m^3 s^-1)
S(t) :: volume of water at time t (in m^3)
P(t) :: precipitation at time t (in mm)
E_pt(t) :: evapotranspiration at time t (in mm)
R(t) :: net rainfall at time t (in mm)

Parameters
K :: constant reaction factor or response factor :: Time :: K > 0


'''

parser = argparse.ArgumentParser(description='Simulate discharge data using the linear reservoir model.')
parser.add_argument("--simulate",dest='simulate', action='store_true',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to true")
parser.add_argument("--no-simulate",dest='simulate', action='store_false',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to false")
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'raw/road_data.csv',
                    help="filename of input dataframe (must end with .csv) (default: %(default)s)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'simulations/linear_reservoir_simulation.csv',
                    help="filename of output dataframe (must end with .csv) (default: %(default)s)")
parser.add_argument("-k", "--k",nargs='?',type=float,default = 0.8,
                    help="constant reaction factor or response factor with unit T (must be positive) (default: %(default)s)")
parser.add_argument("-q", "--q0",nargs='?',type=float,default = 0.01,
                    help="value of discharge (q) at time 0 (must be positive) (default: %(default)s)")
parser.add_argument("-s", "--sigma",nargs='?',type=float,default = 0.5,
                    help="Standard devation of white Gaussian noise N(0,s^2) to be added to discharge (default: %(default)s)")
parser.add_argument("-a", "--catchment_area",nargs='?',type=float,default = 15966.0,
                    help="Area of catchment (in m^2) to be multiplied with discharge (default: %(default)s)")
parser.add_argument("-t", "--tdelta",nargs='?',type=int,default = 1,
                    help="timestep to normalise time by to make the time units seconds (default: %(default)s)")
parser.add_argument("-r", "--randomseed",nargs='?',type=int,default = 22,
                    help="fixed random seed for generating noise (default: %(default)s)")
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

# Print parameters
print(json.dumps(params,indent=2))

''' Generate or read discharge data '''


if args.simulate:
    print('Simulating input data')
    # Simulate rainfall and evapotranspiration data
    rf = np.concatenate((np.random.normal(3, 1, 100) , np.random.normal(3, 0.5, 100)))
    # Some days have no rainfall
    zero_rf_days = random.sample(range(0,len(rf)),30)
    rf[zero_rf_days] = 0

    et = np.concatenate((np.random.normal(2, 1, 100) , np.random.normal(1, 0.5, 100)))
    # Some days have no evapotranspiration
    zero_et_days = random.sample(range(0,len(et)),30)
    et[zero_et_days] = 0

    # Compute net rainfall
    nr = [max(rft - ett,minvalue) for rft, ett in zip(rf, et)]

else:
    print('Reading input data')
    df = pd.read_csv(os.path.join(rd,'data','input',args.input_filename))
    rf = df['rainfall'].values.tolist()
    et = df['evapotranspiration'].values.tolist()
    # Compute net rainfall
    nr = [max(rft - ett,minvalue) for rft, ett in zip(rf, et)]

# Number of time steps
n = len(nr)

# time points
time  = range(0,n)

# Interpolate net_rainfall
nr_int = interp1d(time, nr,fill_value="extrapolate",kind='slinear')

# Define model
def linear_reservoir_model(q,time,nrint):
    return (max(nrint(time),minvalue) - q) * (1./args.k)


# Solve ODE
q = odeint(linear_reservoir_model,args.q0,time,args=(nr_int,))
q_flat = [item*args.fatconv for sublist in q for item in sublist]

# Fix random seed
np.random.seed(args.randomseed)
# Add Gaussian noise to discharge
Q_sim = np.asarray(q_flat).reshape(n,1) + np.random.randn(n,1)*args.sigma
Q_sim = [max(0,qsim[0]) for qsim in Q_sim]

''' Export simulated data to file '''
# .reshape(n,)
df = pd.DataFrame(list(zip(time,rf,et,nr,Q_sim)), columns =['time', 'rainfall','evapotranspiration','net_rainfall', 'discharge'])
print(df.head(10))
df.to_csv(os.path.join(rd,'data','output',args.output_filename),index=False)

print('Done!...')

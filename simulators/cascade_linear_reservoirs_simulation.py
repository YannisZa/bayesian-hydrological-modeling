import pandas as pd
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from sympy.solvers.ode import dsolve
from scipy.interpolate import interp1d
import argparse
import random
import json
import math
import os
#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Simulate discharge data using the linear reservoir model.')
parser.add_argument("--simulate",dest='simulate', action='store_true',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to true")
parser.add_argument("--no-simulate",dest='simulate', action='store_false',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to false")
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'synthetic/rainfall_evapotranspiration_syn.csv',
                    help="filename of input dataframe (must end with .csv)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'cascade_linear_reservoirs_simulation.csv',
                    help="filename of output dataframe (must end with .csv)")
parser.add_argument("-k", "--k",nargs='?',type=float,default = 10.0,
                    help="constant reaction factor or response factor with unit T (must be positive)")
parser.add_argument("-m", "--m",nargs='?',type=int,default = 3,
                    help="number of linear reservoirs to be cascaded")
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

''' Cascade of linear reservoir models

Q_i(t) = R(t) - KdQ_i(t)/dt
Q_i(t) = S_i(t)/K

for 1 <= i <= number of reservoirs (m)

Net rainfal is equal to precipitation minus potential evapotranspiration
R(t) = P(t) - E_pt(t)

'''

# Initilise empty discharge dataframe for each reservoir
q_df = pd.DataFrame(columns=(['time','net_rainfall'] + [('discharge_'+str(i+1)) for i in range(0,m)]))

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

# Store number of rainfall observations
n = len(nr)

# time points
t = range(0,n)

# Store time and net rainfall to dataframe
q_df.loc[:,'time'] = t
q_df.loc[:,'rainfall'] = rf
q_df.loc[:,'evapotranspiration'] = et
q_df.loc[:,'net_rainfall'] = nr

# Interpolate net_rainfall
nr_int = interp1d(t, nr,fill_value="extrapolate")

# Define model
def linear_reservoir_model(q,t,nrint):
    return (nrint(t) - q) * (1./k)

''' Cascade of linear reservoirs '''

# Initialise discharge interpolation
q_int = []
# Loop over reservoirs and propagate discharge
for i in range(0,m):
    if i == 0:
        # Solve ODE
        q = odeint(linear_reservoir_model,q0,t,args=(nr_int,))
    else:
        q0 = q_flat[len(q_flat)-1]
        # Solve ODE
        q = odeint(linear_reservoir_model,q0,t,args=(q_int,))

    # Flatten q
    q_flat = [max(item,0.0) for sublist in q for item in sublist]
    # Populate discharge dataframe
    q_df.loc[:,('discharge_'+str(i+1))] = q_flat
    # Interpolate discharge
    q_int = interp1d(t, q_flat,fill_value="extrapolate",kind='slinear')


# Generate random errors
error = np.random.normal(0,sigma_noise,len(q_flat))
# Add Gaussian noise to discharge
q_flat = [max(q+e,0) for q,e in zip(q_flat,error)]
q_df.loc[:,('discharge_'+str(i+1))] = q_flat

''' Export data to file '''

# Export discharge dataframe to file
filename = os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename)
q_df.to_csv(filename,index=False)

print('Done!...')

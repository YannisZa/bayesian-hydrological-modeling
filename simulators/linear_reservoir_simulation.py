import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from tqdm import tqdm
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
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'simulations/linear_reservoir_simulation.csv',
                    help="filename of output dataframe (must end with .csv)")
parser.add_argument("-k", "--k",nargs='?',type=float,default = 10.0,
                    help="constant reaction factor or response factor with unit T (must be positive)")
parser.add_argument("-q", "--q0",nargs='?',type=float,default = 0.01,
                    help="value of discharge (q) at time 0 (must be positive)")
parser.add_argument("-s", "--sigma_noise",nargs='?',type=float,default = 0.01,
                    help="Standard deviation of white Gaussian noise N(0,s^2) to be added to discharge ")
args = parser.parse_args()

# Flag whether to simulate or use existing data
simulate = args.simulate

# Fixed parameters
k = args.k # constant reaction factor or response factor :: Time :: k > 0

# Let Q(0) = 0 - Discharge at time zero is zero
q0 = args.q0

# Gaussian error noise
sigma_noise = args.sigma_noise

print(json.dumps(vars(args),indent=2))

''' Linear reservoir model

Q(t) = R(t) - KdQ(t)/dt
Q(t) = S(t)/K

Net rainfal is equal to precipitation minus potential evapotranspiration
R(t) = P(t) - E_pt(t)

'''

''' Generate or read data from file '''

if simulate:
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

    # Compute
    nr = [max(rft - ett,0.0001) for rft, ett in zip(rf, et)]

else:
    print('Reading input data')
    df = pd.read_csv(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/input/',args.input_filename))
    rf = df['rainfall'].values.tolist()
    et = df['evapotranspiration'].values.tolist()
    # Compute
    nr = [max(rft - ett,0.0001) for rft, ett in zip(rf, et)]

n = len(nr)

# time points
time  = range(0,n)

# Interpolate net_rainfall
nr_int = interp1d(time, nr,fill_value="extrapolate",kind='slinear')

# Define model
def linear_reservoir_model(q,time,nrint):
    return (nrint(time) - q) * (1./k)


# Solve ODE
q = odeint(linear_reservoir_model,q0,time,args=(nr_int,))
q_flat = [max(item,1e-10) for sublist in q for item in sublist]

# Generate random errors
error = np.random.normal(0,sigma_noise,len(q_flat))
# Add Gaussian noise to discharge
#q_flat = [max(q+e,1e-10) for q,e in zip(q_flat,error)]
q_flat = [q for q,e in zip(q_flat,error)]

''' Export data to file '''

filename = os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',args.output_filename)
df = pd.DataFrame(list(zip(time,rf,et,nr,q_flat)), columns =['time', 'rainfall','evapotranspiration','net_rainfall', 'discharge_approx'])
df.to_csv(filename,index=False)

print('Done!...')

import pandas as pd
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from sympy.solvers.ode import dsolve
from scipy.interpolate import interp1d
from types import SimpleNamespace
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
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'hymod_simulation.csv',
                    help="filename of output dataframe (must end with .csv)")
parser.add_argument("-c", "--cmax",nargs='?',type=float,default = 600.0,
                    help="maximum soil water storage in length units")
parser.add_argument("-b", "--betak",nargs='?',type=float,default = 0.3,
                    help="shape factor of the main soil-water storage tank that represents the degree of spatial variability of the soil-moisture capacity within the catchment")
parser.add_argument("-al", "--alfa",nargs='?',type=float,default = 0.4,
                    help="factor distributing flow between two series of reservoirs")
parser.add_argument("-kf", "--kfast",nargs='?',type=float,default = 10.0,
                    help="fast runoff: constant reaction factor or response factor with unit T (must be positive)")
parser.add_argument("-ks", "--kslow",nargs='?',type=float,default = 10.0,
                    help="slow runoff: constant reaction factor or response factor with unit T (must be positive)")
parser.add_argument("-m", "--m",nargs='?',type=int,default = 3,
                    help="number of linear reservoirs to be cascaded")
parser.add_argument("-q", "--q0",nargs='?',type=float,default = 0.01,
                    help="value of discharge (q) at time 0 (must be positive)")
parser.add_argument("-w", "--w0",nargs='?',type=float,default = 0.01,
                    help="value of water volume (w) at time 0 (must be positive)")
parser.add_argument("-ar", "--area",nargs='?',type=float,default = 1000,
                    help="area of catchment in m^2")
parser.add_argument("-tdelta", "--tdelta",nargs='?',type=int,default = 3600,
                    help="timestep to normalise time by to make the time units seconds")
parser.add_argument("-s", "--sigma_noise",nargs='?',type=float,default = 0.05,
                    help="Standard deviation of white Gaussian noise N(0,s^2) to be added to discharge ")
params = parser.parse_args()


''' Store fixed parameters '''
# Flag whether to simulate or use existing data
simulate = params.simulate
# Gaussian error noise
sigma_noise = params.sigma_noise


# Fixed parameters
cmax = params.cmax     # Maximum soil water storage
betak = params.betak   # Quantifies variability of the soil water storage over the catchment
alfa = params.alfa      # Controls water contribution to fast runoff
kslow = params.kslow     # Constant reaction factor or response factor for slow runoff in linear reservoir
kfast = params.kfast     # Constant reaction factor or response factor for fast runoff in cascade of linear reservoirs
m = params.m           # Number of linear reservoirs fast runoff will go through

''' Store initial conditions '''
q0 = params.q0
w0 = params.w0
area = params.area # in square meters
tdelta = params.tdelta # num of seconds in day/hour etc. depending on data frequency

# Set conversion factor
fatconv = (1. / 1000.0)/tdelta * area

# Initial water volume stored in catchment
w2 = w0
c1 = 0.0
c2 = 0.0
ffast = 0.0
fslow = 0.0
wslow = q0 / ((1./params.kslow) * fatconv)
wfast = np.zeros(params.m)

print(json.dumps(vars(params),indent=2))

''' HYMOD model

C(t)    :: water depth stored in unsaturated locations of the catchment at time t
W(t)    :: volume of water stored in the catchment at time t
E(t)    :: water losses by evapotranspiration

More information about equations please look at https://distart119.ing.unibo.it/albertonew/?q=node/58
'''

''' Obtain or generate net rainfall data '''

# Initilise empty discharge dataframe for each reservoir
q_df = pd.DataFrame(columns=(['time','slow_discharge','fast_discharge','total_discharge']))

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
    df = pd.read_csv(os.path.join('/Users/Yannis/code/fibe2-mini-project/data/input/',params.input_filename))
    rf = df['rainfall'].values.tolist()
    et = df['evapotranspiration'].values.tolist()
    # Compute
    nr = [rft - ett for rft, ett in zip(rf, et)]

# Store number of rainfall observations
n = len(nr)

# time points
time = range(0,n)

# Store time and net rainfall to dataframe
q_df.loc[:,'time'] = time
q_df.loc[:,'rainfall'] = nr
q_df.loc[:,'evapotranspiration'] = et


# Interpolate net_rainfall
nr_int = interp1d(time, nr,fill_value="extrapolate")

''' Define models '''
# Define model
def linear_reservoir_model(q,t,nrint,k):
    return (nrint(t) - q) * (1./k)

def linear_reservoir_simulation(q,t,nrint,k):
    # Solve ODE
    q = odeint(linear_reservoir_model,q,t,args=(nrint,k))
    q_flat = [max(item,0) for sublist in q for item in sublist]

    return q_flat

def cascade_linear_reservoirs_simultation(q,t,nrint,k):
    # Initialise discharge interpolation
    qint = None
    # Loop over reservoirs and propagate discharge
    for i in range(0,n):
        if i == 0:
            # Solve ODE
            q = odeint(linear_reservoir_model,q,t,args=(nrint,k))
        else:
            # Solve ODE
            q = odeint(linear_reservoir_model,q_flat[len(q_flat)-1],t,args=(qint,k))

        # Flatten q
        q_flat = [item for sublist in q for item in sublist]

        # Interpolate discharge
        qint = interp1d(t, q_flat,fill_value="extrapolate")

    return q_flat


''' Run simulation '''

# Initialise water volume, depth, actual evapotranspiration, surface runoff arrays
W = (-1)*np.ones(n)
C = (-1)*np.ones(n)
E = (-1)*np.ones(n)
ER1 = (-1)*np.ones(n)
ER2 = (-1)*np.ones(n)
# Initialise fast and slow discharge
Qfast = np.zeros(n)
Qslow = np.zeros(n)
Q = np.zeros(n)

for t in tqdm(time):

    W[t] = w2

    ''' Compute excess precipitation and evaporation '''
    temp1 = max( 0.0, 1 - W[t] * ((params.betak+1) / params.cmax))
    c1 = params.cmax * (1 - temp1 ** (1 / (params.betak + 1) ))
    c2 = min(c1 + nr[t], params.cmax)

    # Compute surface runoff 1
    ER1[t] = max((nr[t] - cmax + c1),0.0)

    temp2 = max( (1 - C[t]/params.cmax), 0.0)
    w2 = (params.cmax / (params.betak + 1)) * (1 - temp2 ** (params.betak + 1) )

    # Compute surface runoff 2
    ER2[t] = max( (c2 - c1) - (w2 - W[t]), 0.0)

    # Compute water losses by evapotranspiration
    E[t] = (1. - (((params.cmax - c2) / (params.betak + 1.)) / (params.cmax / (params.betak + 1.)))) * et[t]

    # Update water volume accounting for evapotranspiration losses
    w2 = max(w2 - E[t], 0.0)

    ''' Partition ffast and fslow into fast and slow flow component '''
    ffast = params.alfa * ER2[t] + ER1[t]
    fslow = (1.- params.alfa) * ER2[t]

    ''' Route slow flow component with single linear reservoir (kslow) '''
    wslow = (1. - 1./params.kslow) * (wslow + fslow)
    # Store slow discharge at time t
    qslow = ((1./params.kslow) / (1. - 1./params.kslow)) * wslow

    ''' Route fast flow component with m linear reservoirs (kfast) '''
    qfast = 0.0
    for j in range(params.m):
        wfast[j] = (1.- 1./params.kfast) * wfast[j] + (1. - 1./params.kfast) * ffast
        qfast = (1./params.kfast / (1.- 1./params.kfast)) * wfast[j]
        ffast = qfast

    ''' Compute fast, slow and total flow for time t '''
    Qslow[t] = qslow * fatconv
    Qfast[t] = qfast * fatconv
    Q[t] = Qslow[t] + Qfast[t]

# ''' Route slow flow component with single linear reservoir (kslow) '''
# # Interpolate slow runoff
# qslow = interp1d(time, nr,fill_value="extrapolate")
# Qslow = linear_reservoir_simulation(q0,time,qslow,params.kslow)
#
# ''' Route fast flow component with m linear reservoirs (kfast) '''
# qfast = interp1d(time, nr,fill_value="extrapolate")
# Qfast = cascade_linear_reservoirs_simultation(q0,time,qfast,params.kfast)
#
#
# ''' Compute fast, slow and total flow for time t '''
# Q = np.multiply(Qslow,fatconv) + np.multiply(Qfast,fatconv)


# Generate random errors
error = np.random.normal(0,sigma_noise,len(Q))
# Add Gaussian noise to discharge
Q = [max(q+e,0) for q,e in zip(Q,error)]

# Populate q_df with discharges
q_df.loc[:,'fast_discharge'] = Qfast
q_df.loc[:,'slow_discharge'] = Qslow
q_df.loc[:,'total_discharge'] = Q

print(q_df.head(10))


''' Export data to file '''

# Export discharge dataframe to file
filename = os.path.join('/Users/Yannis/code/fibe2-mini-project/data/output/',params.output_filename)
q_df.to_csv(filename,index=False)

print('Done!...')

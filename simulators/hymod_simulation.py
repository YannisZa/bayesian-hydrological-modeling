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
import sys
import os

''' HYMOD model

C(t)    :: water depth stored in unsaturated locations of the catchment at time t
W(t)    :: volume of water stored in the catchment at time t
E(t)    :: water losses by evapotranspiration

ADD HERE...

More information about equations please look at https://distart119.ing.unibo.it/albertonew/?q=node/58
'''


parser = argparse.ArgumentParser(description='Simulate discharge data using the linear reservoir model.')
parser.add_argument("--simulate",dest='simulate', action='store_true',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to true")
parser.add_argument("--no-simulate",dest='simulate', action='store_false',
                    help="sets flag for whether to simulate or use synthetic data for rainfall and evapotranspiration to false")
parser.add_argument("-i", "--input_filename",nargs='?',type=str,default = 'synthetic/rainfall_evapotranspiration_syn.csv',
                    help="filename of input dataframe (must end with .csv)")
parser.add_argument("-o", "--output_filename",nargs='?',type=str,default = 'simulations/hymod_simulation.csv',
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
parser.add_argument("-nr", "--nreservoirs",nargs='?',type=int,default = 3,
                    help="number of linear reservoirs for fast flow to be cascaded")
parser.add_argument("-q", "--q0",nargs='?',type=float,default = 0.01,
                    help="value of discharge (q) at time 0 (must be positive)")
parser.add_argument("-w", "--w0",nargs='?',type=float,default = 0.01,
                    help="value of water volume (w) at time 0 (must be positive)")
parser.add_argument("-s", "--sigma",nargs='?',type=float,default = 0.05,
                    help="Standard deviation of white Gaussian noise N(0,s^2) to be added to discharge ")
parser.add_argument("-a", "--catchment_area",nargs='?',type=float,default = 3600.0,
                    help="Area of catchment (in m^2) to be multiplied with discharge")
parser.add_argument("-t", "--tdelta",nargs='?',type=int,default = 1,
                    help="timestep to normalise time by to make the time units seconds")
parser.add_argument("-r", "--randomseed",nargs='?',type=int,default = 22,
                    help="fixed random seed for generating noise")
args = parser.parse_args()
params = vars(args)

# Get current working directory and project root directory
cwd = os.getcwd()
rd = os.path.join(cwd.split('fibe2-mini-project/', 1)[0])
if not rd.endswith('fibe2-mini-project'):
    rd = os.path.join(cwd.split('fibe2-mini-project/', 1)[0],'fibe2-mini-project')

# Export model parameters
with open(os.path.join(rd,'data','output',args.output_filename.replace('.csv','_true_parameters.json')), 'w') as f:
    json.dump(params, f)

print(json.dumps(params,indent=2))


''' Store initial conditions '''

# Set conversion factor
fatconv = (1. / 1000.0)/args.tdelta * args.catchment_area

# Initial water volume stored in catchment
w2 = args.q0
c1 = 0.0
c2 = 0.0
ffast = 0.0
fslow = 0.0
wslow = args.q0 / ((1./args.kslow) * fatconv)
wfast = np.zeros(args.nreservoirs)


''' Generate or read discharge data '''

# Initilise empty discharge dataframe for each reservoir
q_df = pd.DataFrame(columns=(['time','slow_discharge','fast_discharge','total_discharge']))

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
    nr = [rft - ett for rft, ett in zip(rf, et)]

else:
    print('Reading input data')
    df = pd.read_csv(os.path.join(rd,'data','input',args.input_filename))
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
    temp1 = max( 0.0, 1 - W[t] * ((args.betak+1) / args.cmax))
    c1 = args.cmax * (1 - temp1 ** (1 / (args.betak + 1) ))
    c2 = min(c1 + nr[t], args.cmax)

    # Compute surface runoff 1
    ER1[t] = max((nr[t] - args.cmax + c1),0.0)

    temp2 = max( (1 - C[t]/args.cmax), 0.0)
    w2 = (args.cmax / (args.betak + 1)) * (1 - temp2 ** (args.betak + 1) )

    # Compute surface runoff 2
    ER2[t] = max( (c2 - c1) - (w2 - W[t]), 0.0)

    # Compute water losses by evapotranspiration
    E[t] = (1. - (((args.cmax - c2) / (args.betak + 1.)) / (args.cmax / (args.betak + 1.)))) * et[t]

    # Update water volume accounting for evapotranspiration losses
    w2 = max(w2 - E[t], 0.0)

    ''' Partition ffast and fslow into fast and slow flow component '''
    ffast = args.alfa * ER2[t] + ER1[t]
    fslow = (1.- args.alfa) * ER2[t]

    ''' Route slow flow component with single linear reservoir (kslow) '''
    wslow = (1. - 1./args.kslow) * (wslow + fslow)
    # Store slow discharge at time t
    qslow = ((1./args.kslow) / (1. - 1./args.kslow)) * wslow

    ''' Route fast flow component with m linear reservoirs (kfast) '''
    qfast = 0.0
    for j in range(args.nreservoirs):
        wfast[j] = (1.- 1./args.kfast) * wfast[j] + (1. - 1./args.kfast) * ffast
        qfast = (1./args.kfast / (1.- 1./args.kfast)) * wfast[j]
        ffast = qfast

    ''' Compute fast, slow and total flow for time t '''
    Qslow[t] = qslow * fatconv
    Qfast[t] = qfast * fatconv
    Q[t] = Qslow[t] + Qfast[t]


# Fix random seed
np.random.seed(args.randomseed)
# Add Gaussian noise to discharge
Q_sim = np.asarray(Q).reshape(n,1) + np.random.randn(n,1)*args.sigma

# Populate q_df with discharges
q_df.loc[:,'fast_discharge'] = Qfast
q_df.loc[:,'slow_discharge'] = Qslow
q_df.loc[:,'discharge'] = Q.reshape(n,)


''' Export data to file '''
q_df.to_csv(os.path.join(rd,'data','output',args.output_filename),index=False)
print('Done!...')

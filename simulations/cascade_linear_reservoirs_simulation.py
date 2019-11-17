import pandas as pd
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from sympy.solvers.ode import dsolve
from scipy.interpolate import interp1d
import math
import os
#import matplotlib.pyplot as plt


''' Linear reservoir model

dQ(t)/dt = a(R(t) - Q(t)) * (Q(t))^b
Q(t) = (S(t))^m/K

where
    a = m*(1/K)^(1/m) :: a > 0
    b = (m-1)/m :: -infty < b < 1


Net rainfal is equal to precipitation minus potential evapotranspiration
R(t) = P(t) - E_pt(t)

'''

# Fixed parameters
m = 1 # store exponent :: - :: m > 0
k = 10. # constant reaction factor of each of the linear reservoirs :: Time :: k > 0

# Compute derived parameters
a = m*(1./k)**(1./m)
b = (m-1)/m

# Raise ValueError if m is negative
if m <= 0:
    raise ValueError(f'Number of reservoirs {m} has to be positive.')

# Let Q(0) = 0 - Discharge at time zero is zero
q0 = 0.1

if simulate:
    # Simulate net rainfall data
    nr = np.concatenate((np.random.normal(5, 1, 300) , np.random.normal(0, 1, 300)))
    #nr = nr[net_rainfall>0]
else:
    nr_df = pd.read_csv('/Users/Yannis/code/fibe2-mini-project/data/net_rainfall_sim.csv')
    nr = nr_df['net_rainfall'].values.tolist()

n_samples = len(nr)

# time points
t = range(0,n_samples)
#t = np.linspace(start=0,stop=100000,num=n_samples)#range(0,n_samples)

# Interpolate net_rainfall
nr_int = interp1d(t, nr)

# Define model
def cascade_linear_reservoir_model(q,t,nrint):
    return ( a * ( nrint(t) - q ) * (q**b) )


# Solve ODE
y = odeint(cascade_linear_reservoir_model,q0,t[:-1],args=(nr_int,))
y_flat = [item for sublist in y for item in sublist]
#print(y_flat)

print('Done!...')


filename = '/Users/Yannis/code/fibe2-mini-project/data/cascade_linear_reservoirs_discharge_simulation.csv'
df = pd.DataFrame(list(zip(t,y_flat,nr)), columns =['time', 'discharge','net_rainfall'])
df.to_csv(filename,index=False)

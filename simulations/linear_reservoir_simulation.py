import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os
#import matplotlib.pyplot as plt


''' Linear reservoir model

Q(t) = R(t) - KdQ(t)/dt
Q(t) = S(t)/K

Net rainfal is equal to precipitation minus potential evapotranspiration
R(t) = P(t) - E_pt(t)

'''

# Flag whether to simulate or use existing data
simulate = True

# Fixed parameters
k = 10. # constant reaction factor :: Time :: k > 0

# Let Q(0) = 0 - Discharge at time zero is zero
q0 = 0

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

# Interpolate net_rainfall
nr_int = interp1d(t, nr)

# Define model
def linear_reservoir_model(q,t,nrint):
    return (nrint(t) - q) * (1./k)


# Solve ODE
y = odeint(linear_reservoir_model,q0,t[:-1],args=(nr_int,))
y_flat = [item for sublist in y for item in sublist]

print('Done!...')


filename = '/Users/Yannis/code/fibe2-mini-project/data/linear_reservoir_discharge_simulation.csv'
df = pd.DataFrame(list(zip(t, y_flat,nr)), columns =['time', 'discharge','net_rainfall'])
df.to_csv(filename,index=False)

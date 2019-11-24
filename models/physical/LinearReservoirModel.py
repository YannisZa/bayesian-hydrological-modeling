from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np
import json
import math
import os

n_states = 1
n_odeparams = 1
n_ivs = 1

class LinearReservoirModel(object):

    def __init__(self, n_states, n_odeparams, n_ivs, net_rainfall_data, y0=None):
        self._n_states = n_states
        self._n_odeparams = n_odeparams
        self._n_ivs = n_ivs
        self._y0 = y0
        self._nr = net_rainfall_data

    def simulate(self, parameters, times):
        return self._simulate(parameters, times, self._nr , False)

    def simulate_with_sensitivities(self, parameters, times):
        return self._simulate(parameters, times, self._nr , True)

    def _simulate(self, parameters, times, net_rainfall_data, sensitivities):
        k, q0 = [x for x in parameters]
        # print('k',k)
        # print('q0',q0)

        # Interpolate net_rainfall
        nr_int = interp1d(times, net_rainfall_data,fill_value="extrapolate",kind='slinear')

        # Define model
        def r(q,time,k,nrint):
            return (nrint(time) - q) * (1./k)

        if sensitivities:
            def jac(x):
                ret = np.zeros((self._n_states, self._n_states))
                ret[0, 0] = (-1./k)
                return ret

            def dfdp(x,t,nrint):
                ret = np.zeros((self._n_states,
                                self._n_odeparams + self._n_ivs))
                ret[0, 0] = (-1./(k**2)) * (nrint(t) - x)
                return ret

            def rhs(q_and_dqdp, t, k, nrint):
                q = q_and_dqdp[0:self._n_states]
                dqdp = q_and_dqdp[self._n_states:].reshape((self._n_states,
                                                            self._n_odeparams + self._n_ivs))
                dqdt = r(q, t, k, nrint)

                # print('jacobian',jac(q))
                # print('dqdp',dqdp)
                # print('dfdp',dfdp(q,t,nrint))

                d_dqdp_dt = jac(q)*dqdp + dfdp(q,t,nrint) # CHANGED CODE HERE np.matmul(jac(q), dqdp) + dfdp(q,t,nrint)
                return np.concatenate((dqdt, d_dqdp_dt.reshape(-1)))

            y0 = np.zeros( (n_states*(n_odeparams+n_ivs)) + n_states ) # CHANGED CODE HERE 2*
            y0[2] = 1.            #\frac{\partial  [X]}{\partial Xt0} at t==0, and same below for Y
            y0[0:n_states] = q0
            result = odeint(rhs, y0, times, (k,nr_int),rtol=1e-6,atol=1e-5)
            values = result[:, 0:self._n_states]
            dvalues_dp = result[:, self._n_states:].reshape((len(times),
                                                             self._n_states,
                                                             self._n_odeparams + self._n_ivs))
            return values, dvalues_dp

        else:

            # Solve ODE
            q = odeint(r,q0,times,args=(k,nr_int),rtol=1e-6,atol=1e-5)
            q_flat = [item for sublist in q for item in sublist]

            return q_flat

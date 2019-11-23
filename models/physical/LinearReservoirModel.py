from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np
import json
import math
import os

n_states = 1
n_odeparams = 1
n_ivs = 0

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

        # Interpolate net_rainfall
        nr_int = interp1d(times, net_rainfall_data,fill_value="extrapolate",kind='slinear')

        # Define model
        def r(q,time,k,nrint):
            return (nrint(time) - q) * (1./k)

        if sensitivities:
            def jac(q,t,nrint):
                ret = np.zeros((self._n_states, self._n_states))
                ret[0, 0] = (-1./(k**2)) * (nrint(t) - q)
                return ret

            def dfdp(y):
                X, Y = y
                ret = np.zeros((self._n_states,
                                self._n_odeparams + self._n_ivs))
                ret[0, 0] = (-1./k)
                return ret

            def rhs(q_and_dqdp, t, k, nrint):
                q = q_and_dqdp[0:self._n_states]
                dqdp = q_and_dqdp[self._n_states:]#.reshape((self._n_states,
                                                            #self._n_odeparams + self._n_ivs))
                dqdt = r(q, t, k, nrint)


                jacobian = jac(q,t,nrint)
                print('dqdp',dqdp.shape)
                print('jacobian',jacobian.shape)

                d_dqdp_dt = np.matmul(jac(q,t,nrint), dqdp) + dfdp(q)
                return np.array((dqdt, d_dqdp_dt))#.reshape(-1)))

            y0 = np.zeros( (2*(n_odeparams+n_ivs)) + n_states )
            y0[1] = 1.            #\frac{\partial  [X]}{\partial Xt0} at t==0, and same below for Y
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
            q_flat = [max(item,0) for sublist in q for item in sublist]

            return q_flat

# ode_model = LinearReservoirModel()

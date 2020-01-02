from scipy.interpolate import interp1d
from scipy.integrate import odeint
from tqdm import tqdm
import theano
from theano import *
import theano.tensor as tt
import numpy as np
import pymc3 as pm
import pandas as pd
theano.config.exception_verbosity='high'

class HymodModel(object):

    def __init__(self, rainfall, evapotranspiration, params):
            self._q0 = np.array([params.q0], dtype=np.float64)
            self._times = range(0,len(rainfall))
            self._rn = rainfall#interp1d(self._times, net_rainfall_data,fill_value="extrapolate",kind='slinear')
            self._et = evapotranspiration
            self._n_times = len(rainfall)
            self._n_states = 1

    def _simulate(self, parameters, times, true_arguments):

        cmax,betak,alfa,kfast,kslow = [x for x in parameters]

        # Initial water volume stored in catchment
        w2 = true_arguments.q0
        c1 = 0.0
        c2 = 0.0
        ffast = 0.0
        fslow = 0.0
        wslow = true_arguments.q0 / ((1./kslow) * true_arguments.fatconv)
        wfast = np.zeros(true_arguments.nreservoirs)


        # Number of time steps
        n = len(times)

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

        for t in times: # tqdm(times):

            W[t] = w2

            ''' Compute excess precipitation and evaporation '''
            temp1 = max( 0.0, 1 - W[t] * ((betak+1) / cmax))
            c1 = cmax * (1 - temp1 ** (1 / (betak + 1) ))
            c2 = min(c1 + self._rn[t], cmax)

            # Compute surface runoff 1
            ER1[t] = max((self._rn[t] - cmax + c1),0.0)

            temp2 = max( (1 - C[t]/cmax), 0.0)
            w2 = (cmax / (betak + 1)) * (1 - temp2 ** (betak + 1) )

            # Compute surface runoff 2
            ER2[t] = max( (c2 - c1) - (w2 - W[t]), 0.0)

            # Compute water losses by evapotranspiration
            E[t] = (1. - (((cmax - c2) / (betak + 1.)) / (cmax / (betak + 1.)))) * self._et[t]

            # Update water volume accounting for evapotranspiration losses
            w2 = max(w2 - E[t], 0.0)

            ''' Partition ffast and fslow into fast and slow flow component '''
            ffast = alfa * ER2[t] + ER1[t]
            fslow = (1.- alfa) * ER2[t]

            ''' Route slow flow component with single linear reservoir (kslow) '''
            wslow = (1. - 1./kslow) * (wslow + fslow)
            # Store slow discharge at time t
            qslow = ((1./kslow) / (1. - 1./kslow)) * wslow

            ''' Route fast flow component with m linear reservoirs (kfast) '''
            qfast = 0.0
            for j in range(true_arguments.nreservoirs):
                wfast[j] = (1.- 1./kfast) * wfast[j] + (1. - 1./kfast) * ffast
                qfast = (1./kfast / (1.- 1./kfast)) * wfast[j]
                ffast = qfast

            ''' Compute fast, slow and total flow for time t '''
            Qslow[t] = qslow * true_arguments.fatconv
            Qfast[t] = qfast * true_arguments.fatconv
            Q[t] = Qslow[t] + Qfast[t]

        return np.array([[max(0.0001,qsim)] for qsim in Q]).reshape(len(Q),1)
        #Q.reshape(n,1)

    def simulate(self, params, true_args):
        return self._simulate(params, self._times, true_args)

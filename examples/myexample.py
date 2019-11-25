from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import theano
from theano import *
import pymc3 as pm
import theano.tensor as tt
THEANO_FLAGS='optimizer=fast_compile'
theano.config.exception_verbosity= 'high'
theano.config.floatX = 'float64'

n_states = 1
n_odeparams = 1
n_ivs = 1

class LinearReservoirModel(object):
    def __init__(self, n_states, n_odeparams, n_ivs,net_rainfall_data, y0=None):
        self._n_states = n_states
        self._n_odeparams = n_odeparams
        self._n_ivs = n_ivs
        self._y0 = y0
        self._nr = net_rainfall_data

    def simulate(self, parameters, times):
        return self._simulate(parameters, times, self._nr, False)
    def simulate_with_sensitivities(self, parameters, times):
        return self._simulate(parameters, times, self._nr, True)
    def _simulate(self, parameters, times, net_rainfall_data, sensitivities):
        k, q0 = [x for x in parameters]

        # Interpolate net_rainfall
        nr_int = interp1d(times, net_rainfall_data,fill_value="extrapolate",kind='slinear')

        def r(q,time,k,nrint):
            return (nrint(time) - q) * (1./k)
        if sensitivities:
            def jac(k):
                ret = np.zeros((self._n_states, self._n_states))
                ret[0, 0] = (-1./k)
                return ret

            def dfdp(x,t,k,nrint):
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

                d_dqdp_dt = jac(k)*dqdp + dfdp(q,t,k,nrint) # CHANGED CODE HERE np.matmul(jac(q), dqdp) + dfdp(q,t,nrint)
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
            q = odeint(r,q0,times,args=(k,nr_int),rtol=1e-6,atol=1e-5)
            q_flat = [item for sublist in q for item in sublist]

            return q_flat


q = [0.01, 0.084788051,0.289827287,0.487426902,0.623592162,0.855202214,0.901709887,0.87936577,0.857067839,0.775516564,0.701725939,0.675138958,0.68101658,0.64644605,0.701305112,0.747128907,0.676039744,0.668502137,0.731464651,0.766588801]
nr = [1.618666063,0.0001,4.405308823,0.394073731,3.392555321,2.733285785,0.0001,1.31186209,0.0001,0.0001,0.0001,0.83074128,0.646141131,0.0001,2.405660466,0.0001,0.0001,1.174002978,1.481146447,0.73244669]
times = np.linspace(0,len(q))
nr_int = interp1d(times, nr,fill_value="extrapolate",kind='slinear')
ode_model = LinearReservoirModel(n_states, n_odeparams, n_ivs, nr_int)


class ODEGradop(theano.Op):
    def __init__(self, numpy_vsp):
        self._numpy_vsp = numpy_vsp

    def make_node(self, x, g):
        x = theano.tensor.as_tensor_variable(x)
        g = theano.tensor.as_tensor_variable(g)
        node = theano.Apply(self, [x, g], [g.type()])
        return node

    def perform(self, node, inputs_storage, output_storage):
        x = inputs_storage[0]

        g = inputs_storage[1]
        out = output_storage[0]
        out[0] = self._numpy_vsp(x, g)       # get the numerical VSP

class ODEop(theano.Op):

    def __init__(self, state, numpy_vsp):
        self._state = state
        self._numpy_vsp = numpy_vsp

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)

        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs_storage, output_storage):
        x = inputs_storage[0]
        out = output_storage[0]

        out[0] = self._state(x)               # get the numerical solution of ODE states

    def grad(self, inputs, output_grads):
        x = inputs[0]
        g = output_grads[0]

        grad_op = ODEGradop(self._numpy_vsp)  # pass the VSP when asked for gradient
        grad_op_apply = grad_op(x, g)

        return [grad_op_apply]

class solveCached(object):
    def __init__(self, times, n_params, n_outputs):

        self._times = times
        self._n_params = n_params
        self._n_outputs = n_outputs
        self._cachedParam = np.zeros(n_params)
        self._cachedSens = np.zeros((len(times), n_outputs, n_params))
        self._cachedState = np.zeros((len(times),n_outputs))

    def __call__(self, x):

        if np.all(x==self._cachedParam):
            state, sens = self._cachedState, self._cachedSens

        else:
            state, sens = ode_model.simulate_with_sensitivities(x, times)

        return state, sens

times = np.arange(0, len(q)) # number of measurement points (see below)
cached_solver=solveCached(times, n_odeparams + n_ivs, n_states)

def state(x):
    State, Sens = cached_solver(np.array(x,dtype=np.float64))
    cached_solver._cachedState, cached_solver._cachedSens, cached_solver._cachedParam = State, Sens, x
    return State.reshape((len(State),))

def numpy_vsp(x, g):
    numpy_sens = cached_solver(np.array(x,dtype=np.float64))[1].reshape((n_states*len(times),len(x)))
    return numpy_sens.T.dot(g)



# Define the data matrix
Q = np.vstack((q))


# Now instantiate the theano custom ODE op
my_ODEop = ODEop(state,numpy_vsp)

# The probabilistic model
with pm.Model() as LR_model:

    # Priors for unknown model parameters
    k = pm.Uniform('k', lower=0.01, upper=10)

    # Priors for initial conditions and noise level
    q0 = pm.Lognormal('q0', mu=np.log(1.2), sd=1)
    sigma = pm.Lognormal('sigma', mu=-1, sd=1, shape=1)


    # Forward model
    all_params = pm.math.stack([k,q0],axis=0)
    ode_sol = my_ODEop(all_params)
    forward = ode_sol.reshape(Q.shape)

    # log_forward = pm.math.log(forward)
    # log_forward_print = tt.printing.Print('log_forward')(log_forward.shape)
    # tt.printing.Print('sigma')(sigma.shape)

    # Likelihood
    Q_obs = pm.Lognormal('Q_obs', mu=pm.math.log(forward), sd=sigma, observed=Q)


    print(LR_model.check_test_point())

    # Y_obs_print = tt.printing.Print('Y_obs')(Y_obs)

    trace = pm.sample(n_init=1500, tune=1000, chains=1, init='adapt_diag')

trace['diverging'].sum()

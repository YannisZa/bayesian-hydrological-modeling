from scipy.integrate import odeint
import numpy as np
import theano
from theano import *


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

    def __init__(self, ode_model, times, n_params, n_outputs):
        self._times = times
        self._n_params = n_params
        self._n_outputs = n_outputs
        self._cachedParam = np.zeros(n_params)
        self._cachedSens = np.zeros((len(times), n_outputs, n_params))
        self._cachedState = np.zeros((len(times),n_outputs))
        self._ode_model = ode_model

    def __call__(self, x):

        if np.all(x==self._cachedParam):
            state, sens = self._cachedState, self._cachedSens

        else:
            state, sens = self._ode_model.simulate_with_sensitivities(x, self._times)

        return state, sens

class Run(object):

    def __init__(self, ode_model, times, n_params, n_outputs):
        self.cached_solver = solveCached(ode_model, times, n_params + 0, n_outputs)

    def state(self,x):
        State, Sens = self.cached_solver(np.array(x,dtype=np.float64))
        self.cached_solver._cachedState, self.cached_solver._cachedSens, self.cached_solver._cachedParam = State, Sens, x
        return State.reshape((2*len(State),))

    def numpy_vsp(self,x, g):
        numpy_sens = self.cached_solver(np.array(x,dtype=np.float64))[1].reshape((n_states*len(times),len(x)))
        return numpy_sens.T.dot(g)

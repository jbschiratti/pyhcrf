#icython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
cimport numpy as np
from numpy import ndarray
from numpy cimport ndarray
from numpy import logaddexp, inf
from numpy import exp
#cdef extern from "math.h":
#    double exp(double x)


def forward_backward(ndarray[double, ndim=3] x_dot_parameters,
                     ndarray[double, ndim=3] state_parameters,
                     ndarray[double, ndim=1] transition_parameters,
                     ndarray[long, ndim=2] transitions):
    cdef unsigned int n_time_steps = x_dot_parameters.shape[0]
    cdef unsigned int n_states = state_parameters.shape[1]
    cdef unsigned int n_classes = state_parameters.shape[2]

    n_transitions = transitions.shape[0]

    # Add extra 1 time steps, one for start state and one for end
    forward_table = np.full((n_time_steps + 1, n_states, n_classes), fill_value=-inf, dtype='f64')
    forward_transition_table = np.full((n_time_steps + 1, n_states, n_states, n_classes), fill_value=-inf, dtype='f64')
    forward_table[0, 0, :] = 0.0

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            class_number = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            edge_potential = forward_table[t - 1, s0, class_number] + transition_parameters[transition]
            forward_table[t, s1, class_number] = logaddexp(forward_table[t, s1, class_number],
                                                                 edge_potential + x_dot_parameters[t - 1, s1, class_number])
            forward_transition_table[t, s0, s1, class_number] = logaddexp(forward_transition_table[t, s0, s1, class_number],
                                                                                edge_potential +
                                                                                x_dot_parameters[t - 1, s1, class_number])

    backward_table = np.full((n_time_steps + 1, n_states, n_classes), fill_value=-inf, dtype='f64')
    backward_table[-1, -1, :] = 0.0

    for t in range(n_time_steps - 1, -1, -1):
        for transition in range(n_transitions):
            class_number = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            edge_potential = (backward_table[t + 1, s1, class_number] + x_dot_parameters[t, s1, class_number])
            backward_table[t, s0, class_number] = logaddexp(backward_table[t, s0, class_number],
                                                                  edge_potential + transition_parameters[transition])

    return forward_table, forward_transition_table, backward_table

def dummy():
    pass

def log_likelihood(x, cy, state_parameters, transition_parameters, transitions):
    n_time_steps = x.shape[0]
    n_features = x.shape[1]
    n_states = state_parameters.shape[1]
    n_classes = state_parameters.shape[2]
    cdef ndarray[double, ndim=3] x_dot_parameters = x.dot(state_parameters.reshape(n_features, -1)).reshape((n_time_steps, n_states, n_classes))
    (forward_table,
     forward_transition_table,
     backward_table) = forward_backward(x_dot_parameters,
                                        state_parameters,
                                        transition_parameters,
                                        transitions)
    n_time_steps = forward_table.shape[0] - 1
    n_transitions = transitions.shape[0]
    dstate_parameters = np.zeros_like(state_parameters, dtype='f64')
    dtransition_parameters = np.zeros_like(transition_parameters, dtype='f64')

    class_Z = {c: forward_table[-1, -1, c] for c in range(n_classes)}
    Z = -inf
    for c in range(n_classes):
        Z = logaddexp(Z, forward_table[-1, -1, c])

    for t in range(1, n_time_steps + 1):
        for state in range(n_states):
            for c in range(n_classes):
                alphabeta = forward_table[t, state, c] + backward_table[t, state, c]
                if c == cy:
                    dstate_parameters[:, state, c] += ((exp(alphabeta - class_Z[c]) -
                                                        exp(alphabeta - Z)) * x[t - 1, :])
                else:
                    dstate_parameters[:, state, c] -= exp(alphabeta - Z) * x[t - 1, :]

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            c = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            alphabeta = forward_transition_table[t, s0, s1, c] + backward_table[t, s1, c]
            if c == cy:
                dtransition_parameters[transition] += (exp(alphabeta - class_Z[c]) - exp(alphabeta - Z))
            else:
                dtransition_parameters[transition] -= exp(alphabeta - Z)

    return class_Z[cy] - Z, dstate_parameters, dtransition_parameters

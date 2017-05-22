import numpy as np
cimport numpy as np
from numpy import ndarray
from numpy cimport ndarray
from numpy import logaddexp, inf
from numpy.math cimport logaddexp, INFINITY as inf
cdef extern from "math.h":
    double exp(double x)


def forward_backward(ndarray[double, ndim=3] x_dot_params,
                     ndarray[double, ndim=3] st_params,
                     ndarray[double, ndim=1] tr_params,
                     ndarray[long, ndim=2] transitions):
    cdef unsigned int n_time_steps = x_dot_params.shape[0]
    cdef unsigned int n_states = st_params.shape[1]
    cdef unsigned int n_classes = st_params.shape[2]

    cdef unsigned int n_transitions = transitions.shape[0]

    # Add extra 1 time step for start state
    cdef ndarray[double, ndim = 3] fw_table = np.full((n_time_steps + 1, n_states, n_classes), fill_value=-inf, dtype='float64')
    cdef ndarray[double, ndim = 4] fw_tr_table = np.full((n_time_steps + 1, n_states, n_states, n_classes), fill_value=-inf, dtype='float64')
    fw_table[0, 0, :] = 0.0

    cdef ndarray[double, ndim = 3] bw_table = np.full((n_time_steps + 1, n_states, n_classes), fill_value=-inf, dtype='float64')
    bw_table[-1, -1, :] = 0.0

    cdef unsigned int cls, s0, s1
    cdef int t
    cdef double edge_potential

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            cls = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            edge_potential = fw_table[t - 1, s0, cls] + tr_params[transition]
            fw_table[t, s1, cls] = logaddexp(fw_table[t, s1, cls], edge_potential + x_dot_params[t - 1, s1, cls])
            fw_tr_table[t, s0, s1, cls] = logaddexp(fw_tr_table[t, s0, s1, cls], edge_potential + x_dot_params[t - 1, s1, cls])

    for t in range(n_time_steps - 1, -1, -1):
        for transition in range(n_transitions):
            cls = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            edge_potential = (bw_table[t + 1, s1, cls] + x_dot_params[t, s1, cls])
            bw_table[t, s0, cls] = logaddexp(bw_table[t, s0, cls], edge_potential + tr_params[transition])

    return fw_table, fw_tr_table, bw_table

def dummy():
    pass

def log_lik(x,
            long cy,
            ndarray[double, ndim=3] st_params,
            ndarray[double, ndim=1] tr_params,
            ndarray[long, ndim=2] transitions):
    cdef unsigned int n_time_steps = x.shape[0]
    cdef unsigned int n_features = x.shape[1]
    cdef unsigned int n_states = st_params.shape[1]
    cdef unsigned int n_classes = st_params.shape[2]
    cdef ndarray[double, ndim=3] x_dot_params = x.dot(st_params.reshape(n_features, -1)).reshape((n_time_steps, n_states, n_classes))

    cdef ndarray[double, ndim=3] fw_table
    cdef ndarray[double, ndim=4] fw_tr_table
    cdef ndarray[double, ndim=3] bw_table

    (fw_table, fw_tr_table, bw_table) = forward_backward(x_dot_params,
                                                         st_params,
                                                         tr_params,
                                                         transitions)
    n_time_steps = fw_table.shape[0] - 1
    cdef unsigned int n_transitions = transitions.shape[0]
    cdef ndarray[double, ndim=3] dst_params = np.zeros_like(st_params, dtype='float64')
    cdef ndarray[double, ndim=1] dtr_params = np.zeros_like(tr_params, dtype='float64')

    cdef ndarray[double, ndim=1] class_Z = np.empty((n_classes,))
    cdef double Z = -inf
    cdef unsigned int c
    for c in range(n_classes):
        class_Z[c] = fw_table[-1, -1, c]
        Z = logaddexp(Z, fw_table[-1, -1, c])

    cdef unsigned int t, state, transition, s0, s1
    cdef double alphabeta
    for t in range(1, n_time_steps + 1):
        for state in range(n_states):
            for c in range(n_classes):
                alphabeta = fw_table[t, state, c] + bw_table[t, state, c]
                if c == cy:
                    dst_params[:, state, c] += ((exp(alphabeta - class_Z[c]) - exp(alphabeta - Z)) * x[t - 1, :])
                else:
                    dst_params[:, state, c] -= exp(alphabeta - Z) * x[t - 1, :]

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            c = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            alphabeta = fw_tr_table[t, s0, s1, c] + bw_table[t, s1, c]
            if c == cy:
                dtr_params[transition] += (exp(alphabeta - class_Z[c]) - exp(alphabeta - Z))
            else:
                dtr_params[transition] -= exp(alphabeta - Z)

    return class_Z[cy] - Z, dst_params, dtr_params

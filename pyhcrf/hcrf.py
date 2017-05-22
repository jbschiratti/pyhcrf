# File: hcrf.py
# Author: Dirko Coetsee
# License: GPL
# (Contact me if this is a problem.)
# Date: 13 Augustus 2013
# Updated: 22 July 2015 - almost complete re-write to use sklearn-type
# interface.
#           3 Aug 2015 - Done with new interface.

import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from pyhcrf.algorithms import forward_backward, log_lik
from sklearn.base import BaseEstimator, ClassifierMixin


class Hcrf(BaseEstimator, ClassifierMixin):
    """
    The HCRF model.

    Includes methods for training using LM-BFGS, scoring, and testing, and
    helper methods for loading and saving parameter values to and from file.
    """
    def __init__(self,
                 num_states=2,
                 l2_regularization=1.0,
                 transitions=None,
                 st_param_noise=0.001,
                 tr_param_noise=0.001,
                 optimizer_kwargs=None,
                 sgd_stepsize=None,
                 sgd_verbosity=None,
                 random_seed=0,
                 verbosity=0):
        """
        Initialize new HCRF object with hidden units with cardinality
        `num_states`.
        """
        self.l2_regularization = l2_regularization
        assert(num_states > 0)
        self.num_states = num_states
        self.classes_ = None
        self.st_params = None
        self.tr_params = None
        self.transitions = transitions
        self.st_param_noise = st_param_noise
        self.tr_param_noise = tr_param_noise
        self.optimizer_kwargs = optimizer_kwargs
        self._sgd_stepsize = sgd_stepsize
        self._sgd_verbosity = sgd_verbosity
        self._random_seed = random_seed
        self._verbosity = verbosity

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : List of list of ints. Each list of ints represent a training
        example. Each int in that list
            is the index of a one-hot encoded feature.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        classes = list(set(y))
        num_classes = len(classes)
        self.classes_ = classes
        if self.transitions is None:
            self.transitions = self._create_transitions(num_classes,
                                                        self.num_states)

        # Initialise the parameters
        _, num_features = X[0].shape
        num_transitions, _ = self.transitions.shape
        np.random.seed(self._random_seed)
        if self.st_params is None:
            self.st_params = np.random.standard_normal((num_features,
                                                        self.num_states,
                                                        num_classes)) * \
                             self.st_param_noise
        if self.tr_params is None:
            self.tr_params = np.random.standard_normal(
                num_transitions) * self.tr_param_noise

        initial_param_vector = self._stack_params(self.st_params,
                                                  self.tr_params)
        function_evaluations = [0]

        def objective_function(param_vector, batch_start_index=0,
                               batch_end_index=-1):
            ll = 0.0
            grad = np.zeros_like(param_vector)
            st_params, tr_params = self._unstack_params(param_vector)
            for x, ty in zip(X, y)[batch_start_index: batch_end_index]:
                y_index = classes.index(ty)
                dll, dgrad_state, dgrad_transition = log_lik(x,
                                                             y_index,
                                                             st_params,
                                                             tr_params,
                                                             self.transitions)
                dgrad = self._stack_params(dgrad_state, dgrad_transition)
                ll += dll
                grad += dgrad

            # exclude the bias parameters from being regularized
            params_without_bias = np.array(param_vector)
            params_without_bias[0] = 0
            ll -= self.l2_regularization * np.dot(params_without_bias.T,
                                                  params_without_bias)
            grad = grad.flatten() - 2.0 * self.l2_regularization * \
                params_without_bias

            if batch_start_index == 0:
                function_evaluations[0] += 1
                if self._verbosity > 0 and function_evaluations[0] % \
                        self._verbosity == 0:
                    print('{:10} {:10.2f} {:10.2f}'.format(
                        function_evaluations[0], ll, sum(abs(grad))))
            return -ll, -grad

        # If the stochastic gradient stepsize is defined, do 1 epoch of SGD to
        # initialize the parameters.
        if self._sgd_stepsize:
            total_nll = 0.0
            for i in range(len(y)):
                nll, ngrad = objective_function(initial_param_vector, i, i + 1)
                total_nll += nll
                initial_param_vector -= ngrad * self._sgd_stepsize
                if self._sgd_verbosity > 0:
                    if i % self._sgd_verbosity == 0:
                        print('{:10} {:10.2f} {:10.2f}'.format(
                            i, -total_nll / (i + 1) * len(y), sum(abs(ngrad))))

        if self.optimizer_kwargs is not None:
            self._optimizer_result = fmin_l_bfgs_b(objective_function,
                                                   initial_param_vector,
                                                   **self.optimizer_kwargs)
        else:
            self._optimizer_result = fmin_l_bfgs_b(objective_function,
                                                   initial_param_vector)
        self.st_params, self.tr_params = self._unstack_params(
            self._optimizer_result[0])
        return self

    def predict(self, X):
        """Predict the class for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : List of list of ints, one list of ints for each training example.

        Returns
        -------
        y : iterable of shape = [n_samples]
            The predicted classes.
        """
        pred = [self.classes_[prediction.argmax()] for prediction
                in self.predict_proba(X)]
        return pred

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        y = []
        for x in X:
            n_time_steps, n_features = x.shape
            _, n_states, n_classes = self.st_params.shape
            x_dot_params = x.dot(self.st_params.
                                 reshape(n_features, -1)).\
                reshape((n_time_steps, n_states, n_classes))
            forward_table, _, _ = forward_backward(x_dot_params,
                                                   self.st_params,
                                                   self.tr_params,
                                                   self.transitions)
            # normalize by subtracting log-sum to avoid overflow
            y.append(np.exp(forward_table[-1, -1, :]) / sum(np.exp(
                forward_table[-1, -1, :])))
        return np.array(y)

    @staticmethod
    def _create_transitions(num_classes, num_states):
        # 0    o>
        # 1    o>\\\
        # 2   /o>/||
        # 3  |/o>//
        # 4  \\o>/
        transitions = []
        for c in range(num_classes):  # The zeroth state
            transitions.append([c, 0, 0])
        for state in range(0, num_states - 1):  # Subsequent states
            for c in range(num_classes):
                # To the next state
                transitions.append([c, state, state + 1])
                # Stays in same state
                transitions.append([c, state + 1, state + 1])
                if state > 0:
                    # From the start state
                    transitions.append([c, 0, state + 1])
                if state < num_states - 1:
                    # To the end state
                    transitions.append([c, state + 1, num_states - 1])
        transitions = np.array(transitions, dtype='int64')
        return transitions

    @staticmethod
    def _stack_params(st_params, tr_params):
        return np.concatenate((st_params.flatten(), tr_params))

    def _unstack_params(self, param_vector):
        st_param_shape = self.st_params.shape
        num_st_params = np.prod(st_param_shape)
        return param_vector[:num_st_params].reshape(st_param_shape), \
            param_vector[num_st_params:]

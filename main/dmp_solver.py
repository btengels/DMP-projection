"""dmp_solver.py

This file contains the dmp_solver class, which inherits the parameters/methods
from the dmp_model class and adds additional methods for solving for optimal 
policy functions. The model is solved using a nonlinear solver in on the 
model's optimality conditions.

Example:

    >>> model = dmp_solver()  # assumes parameter values given in dmp_model
    >>> model.solve_model()  # solves the model for optimal policy function

    >>> # retrieve the results
    >>> aE = model.aE

If the model is modified in a way that introduces additional policy functions,
the `_residual` method will need to be changed, as well as other methods 
which make and reshape guesses for policy functions being passed into the 
solver.     
"""
import numpy as np
import scipy as sp
import utils 
from scipy.interpolate import bisplrep, bisplev
from dmp_model import dmp_model


class dmp_solver(dmp_model):
    """Class for solving for the model policy function. """

    def solve_model(self):
        """Finds policy functions that minimize the '_residual' function

        """
        # test residual function before throwing into solver (for full traceback)
        self._get_initial_conditions()
        self._get_splines()
        err = self._residual(self.initial_guess)

        # solve residual function, assign key elements of "results" dictionary
        self.results = sp.optimize.fsolve(self._residual, self.initial_guess, full_output=1)
        self.aE = self.results[0].reshape(self.nn, self.nx)
        self.model_residuals_ = self.results[1]['fvec']        

    def _residual(self, aE_vec):
        """Given a policy function, solves the model's optimality condition and returns the error

        An optimal policy function will return an error close to zero. This method will typically 
        be called within a numerical solver to find the optimal policy function. 

        Args:
            aE_vec (np.array): 1 dimensional array with as many elements as the policy function(s)

        Returns:
            numpy array of 1 dimension representing optimality equation errors
        """
        # reshape input (candidate solution) to shape of policy function, assign to self.aE
        self._unpack_residual_inputs(aE_vec)
        self._get_splines()

        # back out current model variables based on policy function
        q = self._get_q(self.aE)
        theta = self._get_theta(q)
        V = self._get_V(theta, self.N)

        # employment tomorrow
        N_p = self._get_N_tplusone(self.N, q, V)

        # interpolate for tomorrow's policy function (will need to check this to see if it holds w/ log consumption)
        func = sp.interpolate.interp2d(self.X, self.N, self.aE) 
        E_p = np.array([func(self.X[i_n], N_p[:, i_n]) for i_n in range(self.nx)])
        E_p = E_p[:, :, 0].T

        # back out future model variables based on policy function
        q_p = self._get_q(E_p)
        theta_p = self._get_theta(q_p)
        lambda_p = self._get_lambda(E_p, q_p)
        gamma_p = self.gamma + self.c*q_p
        w_p = self._get_W(self.X, theta_p, q_p)

        # error in the optimal job creation condition (inside the expectation operator)
        inside_p = self.beta*(self.X - w_p + (1 - self.s)*(gamma_p/q_p - lambda_p))

        # expectations over future states
        rhs = np.dot(inside_p, self.P)
        err = self.aE - rhs
        return err.flatten()

    def _get_initial_conditions(self):
        """Initializes policy functions. Assigns policy functions as attributes to DMP_Model class.

        """
        self.aE = np.linspace(0, 1, self.nn)[:, np.newaxis] * np.linspace(.7, 1.3, self.nx)[np.newaxis]
        self.initial_guess = self.aE.flatten()    

    def _get_splines(self):
        """Gets spline knots based on discrete state space and policy functions

        """
        self.aE_spline = bisplrep(self.Xmesh, self.Nmesh, self.aE)        

    def _unpack_residual_inputs(self, aE_vec):
        """Takes a vector and reshapes it to the size of the policy function(s). 

        Used in `_residual` since any solver takes only a flat vector as an input.

        Args:
            aE_vec (np.array): 1 dimensional array with as many elements as the policy function(s)
        """
        ix_aE = np.prod(self.aE.shape)
        self.aE = aE_vec[0:ix_aE].reshape(self.nn, self.nx)     


if __name__ == "__main__":

    # check parameterization routines
    model = dmp_model()
    solver = dmp_solver() 
    solver._get_initial_conditions()
    solver._get_splines()
    err_before = solver._residual(solver.initial_guess)
    solver.solve_model() 

    err_after = solver._residual(solver.initial_guess)


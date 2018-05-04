"""dmp_model.py

This file contains the dmp_model class, which contains the parameters
and equations which characterize the model. 

This class is not intended to be initialized by itself. Currently it
is inherited by classes for solving/simulating the model.
"""
import numpy as np
import scipy as sp
import utils 
import time
import math
from scipy.interpolate import bisplrep, bisplev, CubicSpline, interp2d


class dmp_model(object):
    """Solves and simulates a DMP-like model using projections methods."""

    def __init__(self):
        """Initializes DMP_Model object. Sets initial parameters using model_params and grid_builder classes

        """
        self.s = .035                           # separation rate
        self.nu_L = 1.25                        # elasticity of Den Haan et al matching function
        self.alpha_L = .50                      # worker's bargaining weight
        self.r_annual = 0.04                    # mean annual discount rate
        self.beta = np.exp(-self.r_annual/12)   # time-discount factor adjusted to match the mean discount rate in the data
        self.eta_L = .5                         # elasticity of the Cobb-Douglas matching function
        self.chi_L = .54                        # level parameter of the Cobb-Douglas matching function
        self.gamma = .3 #0.2581                       # the cost of vacancy posting

        #put these out of the function
        self.z = .71        # value of unemployment benefit
        self.c = 0          # fixed hiring cost
        self.phi = 0        # probability of negotiation breakdown
        self.zeta = 0       # firm's cost of delaying

        # grids
        self.x_bar = 0          # mean technology shock 
        self.rhox = .95**(1/3)  # technology persistence parameter
        self.stdx = .00625      # technology standard deviation        
        self.nx = 15            # the number of grid points for x a column vector

        self.nn = 10            # number of elements in splines = number of N grid points
        self.Nmin = 0.2
        self.Nmax = .98

        self.aE = None
        self.get_grids()

    def _get_E(self, N_t, X_t):
        """Evaluates policy function at a given point in the state space

        Args:
            N_t (float): current employment. Bounded between 0 and 1
            discrete_state (int): index for the discrete state

        Returns:
            float indicating the value of the policy function 
        """
        interp_func = interp2d(self.X, self.N, self.aE)
        E_t = interp_func(X_t, N_t)
        return E_t

    def _get_N_tplusone(self, N_t, q_t, V_t):
        """Finds employment tomorrow from current employment, vacancies, and fill rates 

        Args:
            N_t (float): current employment. Bounded between 0 and 1
            q_t (float): probability of filling vacancy today
            V_t (float): current number of vacancies. 

        Returns:
            float indicating employment tomorrow
        """
        N_tp = (1 - self.s)*N_t + q_t*V_t
        N_tp = np.minimum(self.Nmax, np.maximum(self.Nmin, N_tp))
        return N_tp

    def _get_V(self, theta_t, N_t):
        """Finds vacancy rate from labor market tightness and employment

        Args:
            N_t (float): current employment. Bounded between 0 and 1
            theta_t (float): ratio of vacancy rate to unemployment

        Returns:
            float indicating current vacancy rate
        """
        V_t = theta_t*(1 - N_t)
        return V_t

    def _get_theta(self, q_t):
        """Finds labor market tightness given matching function and vacancy-filling probability

        Args:
            q_t (float): probability of filling vacancy today 

        Returns:
            float indicating labor market tightness
        """
        theta = (q_t**(-self.nu_L) - 1)**(1/self.nu_L)
        return theta

    def _get_q(self, E_t):
        """Finds vacancy fill rate from policy-function/optimality condition

        Args:
            E_t (float): policy function value

        Returns:
            float indicating probability of filling vacancy today
        """
        q = self.gamma/(E_t - self.c)
        q = np.minimum(q, 1)
        q = np.where(q < 0, 1, q)
        return q

    def _get_lambda(self, E, q):
        """
        """
        lambda_t = np.zeros((self.nn, self.nx))
        lambda_t[q==1] = self.gamma + self.c - E[q==1]        
        return lambda_t

    def _get_f(self, theta_t, q_t):
        """Finds the matching probability for unemployed

        Args: 
            theta_t (float or array): ratio of vacancy rate to unemployment
            q_t (float or array): probability of filling vacancy today 

        Returns:
            float or array indicating match probability of unemployed
        """
        f_t = theta_t * q_t
        return f_t

    def _get_W(self, X_t, theta_t, q_t):
        """Finds optimal Wage 

        Args:
            X_t (float or array): productivity
            theta_t (float or array): ratio of vacancy rate to unemployment
            q_t (float or array): probability of filling vacancy today 

        Returns:
            float or array for Nash-bargained wage
        """
        W_t = self.alpha_L*X_t + self.alpha_L*theta_t*(self.gamma + q_t*self.c) + (1 - self.alpha_L)*self.z
        return W_t

    def get_grids(self):
        """Gets all grids and assigns them as grid_builder attributes

        """
        self.P, ln_x = self.get_rouwen_matrix(self.rhox, self.stdx, self.nx)   
        self._get_X_grid(ln_x)
        self._get_N_grid()
        self.Xmesh, self.Nmesh = np.meshgrid(self.X, self.N)

    def _get_X_grid(self, ln_x):
        """Gets X grid from vector ln_x, assigns as attribute with min and max

        Args:
            ln_X (np.array): vector for X in log space
        """
        self.X = np.exp(ln_x)
        self.xmin = min(ln_x)
        self.xmax = max(ln_x)

    def _get_N_grid(self):
        """Gets N grid using compecon spline routines

        """
        self.N = np.linspace(self.Nmin, self.Nmax, self.nn)[:, np.newaxis]
        self.N = np.array([0.2000, 0.2371, 0.3114, 0.4229, 0.5343, 0.6457, 0.7571, 0.8686, 0.9429, 0.9800])[:, np.newaxis]

        # self.fspace = fundefn('spli', self.nn, self.Nmin, self.Nmax)
        # self.N = funnode(fspace)

    def get_rouwen_matrix(self, rho, std, n_states):
        """Gets Rouwenhorst transition matrix with accompanying state space

        Args: 
            rho (float): persistance of AR(1) process
            std (float): standard deviation of underlying shocks
            n_states (int): size of grid for approximating state space

        Returns:
            np.array (n_states, n_states): transition probability matrix
            np.array (n_states,): state space
        """
        step_size = 2*std/math.sqrt((1 - rho**2)*(n_states-1))
        P, ln_grid = utils.rouwen(rho, 0, step_size, n_states)
        return P, ln_grid

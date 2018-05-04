"""dmp_model.py

TODO:   try increasing alphaL
        try decreasing nu_C
        try increasing alphaC

"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import utils 
import time
import math
from scipy.interpolate import bisplrep, bisplev, CubicSpline, interp1d


class dmp_model(object):
    """Solves and simulates a DMP-like model using projections methods."""

    def __init__(self):
        """Initializes DMP_Model object. Sets initial parameters using model_params and grid_builder classes

        """
        self.s_L = .035                         # labor_market separation rate        
        self.nu_L = 1.12                        # elasticity of Den Haan et al matching function
        self.alpha_L = .3                      # worker's bargaining weight
        # self.r_annual = 0.04                    # mean annual discount rate
        self.beta = .997                        # time-discount factor adjusted to match the mean discount rate in the data
        self.eta_L = .5                         # elasticity of the Cobb-Douglas matching function
        self.chi_L = .54                        # level parameter of the Cobb-Douglas matching function
        self.gamma = .125 #0.2581                 # the cost of vacancy posting

        self.s_C = .01/3                        # credit market: separation rate  
        self.nu_C = 1.2                         # credit market: elasticity of Den Haan et al matching function
        self.alpha_C = .3                       # credit market: creditor's bargaining weight
        self.eta_C = .5                         # elasticity of the Cobb-Douglas matching function
        self.chi_C = .54                        # level parameter of the Cobb-Douglas matching function        

    
        #put these out of the function
        self.z = .8             # value of unemployment benefit
        self.c = 0                # fixed hiring cost
        self.phi = 0              # probability of negotiation breakdown
        self.zeta = 0             # firm's cost of delaying
        self.r = 1/self.beta - 1  # risk free rate
        self.kappa_I = .16       # cost of credit market search

        # grids
        self.x_bar = 0            # mean technology shock 
        self.rho_x = .95**(1/3)   # technology persistence parameter
        self.std_x = .0065       # technology standard deviation        
        self.nx = 15            # the number of grid points for x a column vector

        self.kappaB_bar = self.kappa_I           # mean technology shock 
        self.rho_kappaB = .95  # technology persistence parameter
        self.std_kappaB = .006        # technology standard deviation        
        self.n_kappaB = 15            # the number of grid points for x a column vector

        self.nn = 11            # number of elements in splines = number of N grid points
        self.Nmin = 0.2
        self.Nmax = .97

        self.aE = None
        self.get_grids()

    def _get_E(self, N_t, state_index):
        """Evaluates policy function at a given point in the state space

        Args:
            N_t (float): current employment. Bounded between 0 and 1
            discrete_state (int): index for the discrete state

        Returns:
            float indicating the value of the policy function 
        """
        interp_func = interp1d(self.N.flatten(), self.aE[:, state_index])
        E_t = interp_func(N_t)
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
        N_tp = (1 - self.s_C)*((1 - self.s_L)*N_t + q_t*V_t)
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

    def _get_q(self, E_t, K_t):
        """Finds vacancy fill rate from policy-function/optimality condition

        Args:
            E_t (float): policy function value

        Returns:
            float indicating probability of filling vacancy today
        """
        q = (self.gamma + K_t)/E_t
        q = np.minimum(q, 1)
        q = np.where(q < 0, 1, q)
        return q

    def _get_lambda(self, E, q):
        """
        """
        lambda_t = np.zeros((self.nn, self.nx*self.n_kappaB))
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

    def _get_W(self, X_t, K_t, theta_t):
        """Finds optimal Wage 

        Args:
            X_t (float or array): productivity
            theta_t (float or array): ratio of vacancy rate to unemployment
            q_t (float or array): probability of filling vacancy today 

        Returns:
            float or array for Nash-bargained wage
        """
        W_t = self.alpha_L*(X_t + theta_t*(self.gamma/(1 - self.s_C) + (self.r + self.s_C)/((1-self.s_C)*(1 + self.r))*K_t) - (self.r + self.s_C)/(1+self.r)*K_t) + (1 - self.alpha_L)*self.z 
        return W_t

    def _get_Omega(self, K_t, q_t):
        """
        """
        Omega_p = K_t + self.gamma - (1 - self.s_C)/(1 + self.r)*(1 - q_t)*K_t
        return Omega_p

    def _get_Fg(self, X_t, K_t, W_t, Omega_t, q_t):
        """
        """
        Fg_t = X_t - W_t + (1 - self.s_C)*self.s_L*K_t + (1 - self.s_L)*Omega_t/q_t
        return Fg_t

    def _get_Rspread(self, N_p, q, state_index):
        """
        """
        # interpolate for tomorrow's policy function (will need to check this to see if it holds w/ log consumption)
        E_p = np.zeros_like(self.X_grid)
        for i_state in range(len(self.X_grid)):
            func = sp.interpolate.interp1d(self.N.flatten(), self.aE[:, i_state]) 
            E_p[i_state] = func(N_p)          

        # back out future model variables based on policy function
        q_p = self._get_q(E_p, self.K_grid)
        theta_p = self._get_theta(q_p)
        w_p = self._get_W(self.X_grid, self.K_grid, theta_p)
        # Omega_p = self._get_Omega(self.K_grid, q_p)        
        # Fg_p = self._get_Fg(self.X_grid, self.K_grid, w_p, Omega_p, q_p)

        # pc = perfect competition
        Psi_p = self.alpha_C*(self.X_grid - w_p ) + (1 - self.alpha_C)*( (self.gamma/q)*((1 + self.r)/(1 - self.s_C)) - (1 - self.s_L)*self.gamma/q_p)
        Psi_PC_p  = ((self.gamma*(1 + self.r))/(q*(1 - self.s_C)) - (1 - self.s_L)*self.gamma/q_p)

        E_psi_p = np.dot(Psi_p, self.P[:, int(state_index)])
        E_psi_PC_p = np.dot(Psi_PC_p, self.P[:, int(state_index)])


        Rspread_search = q*E_psi_p/self.gamma - (self.s_C + (1 - self.s_C)*self.s_L) - self.r
        Rspread_PC = q*E_psi_PC_p/self.gamma - (self.s_C + (1 - self.s_C)*self.s_L)
        Rspread = Rspread_search - Rspread_PC
        return Rspread

    def get_grids(self):
        """Gets all grids and assigns them as grid_builder attributes

        """
        self.Px, self.ln_x = self.get_rouwen_matrix(self.rho_x, self.std_x, self.nx)
        self.Pk, self.ln_kappaB = self.get_rouwen_matrix(self.rho_kappaB, self.std_kappaB, self.n_kappaB)
        self.P = np.kron(self.Px, self.Pk)

        self._get_exogenous_grids()
        self._get_N_grid()

    def _get_exogenous_grids(self):
        """Gets X grid from vector ln_x, assigns as attribute with min and max

        Args:
            ln_X (np.array): vector for X in log space
        """
        self.X = np.exp(self.ln_x)
        self.xmin = min(self.ln_x)
        self.xmax = max(self.ln_x)

        self.kappa_B = self.kappaB_bar*np.exp(self.ln_kappaB)
        self.K = self._get_K_from_kappaB(self.kappa_B)  # total cost of credit search
        self.kmin = min(self.K)
        self.kmax = max(self.K)

        k, x = np.meshgrid(self.K, self.X)
        self.X_grid = x.flatten()
        self.K_grid = k.flatten()

    def _get_K_from_kappaB(self, kappa_B):
        """
        """
        phi = (kappa_B/self.kappa_I)*(1 - self.alpha_C)/self.alpha_C    # credit market tightness
        p = (1 + phi**self.nu_C)**(-1/self.nu_C)
        K = self.kappa_I/p + kappa_B/(phi * p) 
        return K
 
    def _get_N_grid(self):
        """Gets N grid using compecon spline routines

        """
        self.N = np.linspace(self.Nmin, self.Nmax, self.nn)[:, np.newaxis]
        # self.N = np.array([0.2000, 0.2371, 0.3114, 0.4229, 0.5343, 0.6457, 0.7571, 0.8686, 0.9429, 0.9800])[:, np.newaxis]

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


if __name__ == '__main__':
    dmp = dmp_model()
    dmp._get_exogenous_grids()
    print(dmp.X_grid)
    print(dmp.K_grid)
    


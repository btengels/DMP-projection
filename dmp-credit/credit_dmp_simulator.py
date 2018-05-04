"""dmp_simulator.py

This file contains the dmp_simulator class, which inherits the parameters/methods
from the dmp_model class and adds additional methods for simulating the dmp model
given a set of policy functions. 

Example:

    >>> simulator = dmp_simulator()  # assumes parameters and model equations in dmp_model 
    >>> simulator.simulate_model(continuous=False)  # simulates model on discretized state space
    >>> moments = simulator._get_mean_simu_moments()  # computes model moments from simulated data 

    >>> # object contains raw simulation data
    >>> simulator.X_t  # raw data from simulation (productivity)

If the model is modified in a way that introduces additional state variables some methods
may require changing.    
"""
import numpy as np
import utils 
import scipy.stats as stats
from dmp_model import dmp_model
from dmp_solver import dmp_solver



class dmp_simulator(dmp_model):        
    """Class for simulating series and retreiving moments and/or irfs."""

    def simulate_model(self, continuous=False, solve=True):
        """For a given policy function and set of random shocks, simulates time series 

        Args:
            continuous (boolean): If true, simulate using continuous state space

        """
        # solve model if needed
        if solve:
            self._solve_model()

        # simulate shocks
        sim = utils.simulate_AR1(continuous=continuous)
        if continuous:
            sim.get_shocks()
        else:
            sim.get_shocks()

        # simulate model variables
        U_sims = []
        V_sims = []
        q_sims = []
        theta_sims = []
        rspread_sims = []
        X_sims = []
        K_sims = []

        if sim.continuous:
            ln_X_full = sim.simulate_continuous(self.rho_x, self.x_bar, self.std_x)
            X_full = np.exp(ln_X_full)

            ln_kappaB_full = sim.simulate_continuous(self.rho_kappaB, self.kappaB_bar, self.std_kappaB)
            kappaB_full = np.exp(ln_kappaB_full)
            K_full = self._get_K_from_kappaB(kappaB_full)

        else:
            _, state_index_full = sim.simulate_discrete(self.P, self.X_grid)
            X_full = self.X_grid[state_index_full]
            K_full = self.K_grid[state_index_full]

        for pn in range(sim.parnsim):
            N_t, V_t, q_t, theta_t, rspread_t = self._simulate_series(state_index_full[:, pn])
        
            # remove burnin period from simulation
            N_t = N_t[sim.burnin:-1]
            V_t = V_t[sim.burnin:]
            q_t = q_t[sim.burnin:]
            theta_t = theta_t[sim.burnin:]
            rspread_t = rspread_t[sim.burnin:]
            X_t = X_full[sim.burnin:, pn]
            K_t = K_full[sim.burnin:, pn]

            # save results across loops
            X_sims.append(X_t)
            K_sims.append(K_t)
            U_sims.append(1 - N_t)
            V_sims.append(V_t)
            q_sims.append(q_t)
            theta_sims.append(theta_t)
            rspread_sims.append(rspread_t)

        # Reformat variables (think about whether to do fewer-longer simulations or many-shorter simulations)
        self.U_t = np.array(U_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.V_t = np.array(V_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.X_t = np.array(X_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.K_t = np.array(K_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.q_t = np.array(q_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.theta_t = np.array(theta_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.rspread_t = np.array(rspread_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        
        # back out additional variables f and W
        self.f_t = self._get_f(self.theta_t, self.q_t)
        self.W_t = self._get_W(self.X_t, self.K_t, self.theta_t)

    def _simulate_series(self, state_index, init_conditions=.92):
        """Given a series of state variables, finds correct series of other variables

        Args:
            X_t (numpy array): time series of exogenous variable
            init_conditions (float): initial level of employment; defaults to .92

        Returns:
            numpy arrays for employment, vacancies, vacancy fill rates, and market tightness
        """
        #preallocate empty matrices
        N_t = np.zeros(len(state_index) + 1)
        V_t = np.zeros(len(state_index))
        q_t = np.zeros_like(V_t)
        E_t = np.zeros_like(V_t)
        theta_t = np.zeros_like(V_t)
        rspread_t = np.zeros_like(V_t)
        N_t[0] = init_conditions

        # get model variables for each period
        for t in range(len(state_index)):  
            K_t = self.K_grid[state_index[t]]
            E_t[t] = self._get_E(N_t[t], state_index[t])
            q_t[t] = self._get_q(E_t[t], K_t)  
            theta_t[t] = self._get_theta(q_t[t])
            V_t[t] = self._get_V(theta_t[t], N_t[t])            
            N_t[t + 1] = self._get_N_tplusone(N_t[t], q_t[t], V_t[t])
            
            rspread_t[t] = self._get_Rspread(N_t[t + 1], q_t[t], state_index[t])            

        return N_t, V_t, q_t, theta_t, rspread_t           


    def _get_mean_simu_moments(self):
        """Gets moments from each simulation, returns moment averages across simulations

        Returns:
            numpy array of moments averaged across many time series
        """
        moment_sim_list = []
        for ix in range(self.X_t.shape[1]):
            moment_sim = self._get_simu_moments_once(self.X_t[:, ix], self.U_t[:, ix], self.V_t[:, ix], self.rspread_t[:, ix])                                                                   
            moment_sim_list.append(moment_sim)   

        # Compute mean of moments over all simulations
        mean_moments = np.array(moment_sim_list).mean(axis=0)
        return mean_moments

    def _get_simu_moments_once(self, X_t, U_t, V_t, R_t, monthly=True):
        """Computes moments from several time series

        Args:
            X_t (numpy array): productivity time series
            U_t (numpy array): unemployment time series
            V_t (numpy array): vacancy time series

        Returns:
            numpy array of moments (means, std, auto, corr coefs)
        """
        if monthly:
            # get quarterly averages
            X_q = utils._get_quarterly_avg(X_t)
            U_q = utils._get_quarterly_avg(U_t)
            V_q = utils._get_quarterly_avg(V_t)
            R_q = utils._get_quarterly_avg(R_t)
            
        else:
            X_q = X_t
            U_q = U_t
            V_q = V_t
            R_q = R_t

        # define theta
        theta_q = V_q/U_q
        
        # get hp-filtered series
        X_hp = utils.get_detrended_simulation(X_q)
        U_hp = utils.get_detrended_simulation(U_q)
        V_hp = utils.get_detrended_simulation(V_q)
        theta_hp = utils.get_detrended_simulation(theta_q)
        R_hp = utils.get_detrended_simulation(R_q)
        
        # get difference between log-series and log-hptrend
        X_d = utils.get_log_hpdif_simulation(X_q)
        U_d = utils.get_log_hpdif_simulation(U_q)
        V_d = utils.get_log_hpdif_simulation(V_q)
        theta_d = utils.get_log_hpdif_simulation(theta_q)
        R_d = utils.get_log_hpdif_simulation(R_q)
        
        # get first difference of log-series
        X_fd = utils.get_log_firstdif_simulation(X_q)
        U_fd = utils.get_log_firstdif_simulation(U_q)
        V_fd = utils.get_log_firstdif_simulation(V_q)
        theta_fd = utils.get_log_firstdif_simulation(theta_q)   
        R_fd = utils.get_log_firstdif_simulation(R_q)   

        # use pct-change HP series if log returns NaNs
        if np.isnan(np.std(V_d)):
            theta_d = theta_hp
            V_d = V_hp

        if np.isnan(np.std(R_d)):
            R_d = R_hp
 
        # moment table includes means, 0
        mean_sim = np.array([U_q, V_q, theta_q, X_q, R_q]).mean(axis=1)        
        std_sim = np.array([U_d, V_d, theta_d, X_d, R_d]).std(axis=1, ddof=1)
        skew_sim = stats.skew([U_d, V_d, theta_d, X_d, R_d], axis=1)
        kurt_sim = stats.kurtosis([U_d, V_d, theta_d, X_d, R_d], axis=1, fisher=False)

        autocorrealtion_sim = np.array([utils.AutoCorr(series) for series in [U_fd, V_fd, theta_fd, X_fd, R_fd]], ndmin=2).T
        corrcoef_sim = np.corrcoef(np.array([U_d, V_d, theta_d, X_d, R_d]))
        
        return np.vstack((mean_sim, std_sim, skew_sim, kurt_sim, autocorrealtion_sim, corrcoef_sim))


    def _solve_model(self):
        """
        """
        solver = dmp_solver()
        solver.alpha_L = self.alpha_L
        solver.alpha_C = self.alpha_C
        solver.nu_L = self.nu_L
        
        solver.solve_model() 
        self.aE = solver.aE

if __name__ == "__main__":

    # initialize class and simulate. Note, this uses "initial guess" policy function
    simulator = dmp_simulator()    
    simulator.simulate_model(continuous=False)
    moments = simulator._get_mean_simu_moments()
    print(moments)
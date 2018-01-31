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
        X_sims = []

        if sim.continuous:
            ln_X_full = sim.simulate_continuous(self.rhox, self.x_bar, self.stdx)
            X_full = np.exp(ln_X_full)

        else:
            X_full, index_full = sim.simulate_discrete(self.P, self.X)

        for pn in range(sim.parnsim):
            N_t, V_t, q_t, theta_t = self._simulate_series(X_full[:, pn])

            # remove burnin period from simulation
            N_t = N_t[sim.burnin:-1]
            V_t = V_t[sim.burnin:]
            q_t = q_t[sim.burnin:]
            theta_t = theta_t[sim.burnin:]
            X_t = X_full[sim.burnin:, pn]

            # save results across loops
            X_sims.append(X_t)
            U_sims.append(1 - N_t)
            V_sims.append(V_t)
            q_sims.append(q_t)
            theta_sims.append(theta_t)

        # Reformat variables (think about whether to do fewer-longer simulations or many-shorter simulations)
        self.U_t = np.array(U_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.V_t = np.array(V_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.X_t = np.array(X_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.q_t = np.array(q_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T
        self.theta_t = np.array(theta_sims).reshape([sim.nsim*sim.parnsim, sim.len_sim]).T

        # back out additional variables f and W
        self.f_t = self._get_f(self.theta_t, self.q_t)
        self.W_t = self._get_W(self.X_t, self.theta_t, self.q_t)


    def _simulate_series(self, X_t, init_conditions=.92):
        """Given a series of state variables, finds correct series of other variables

        Args:
            X_t (numpy array): time series of exogenous variable
            init_conditions (float): initial level of employment; defaults to .92

        Returns:
            numpy arrays for employment, vacancies, vacancy fill rates, and market tightness
        """
        #preallocate empty matrices
        N_t = np.zeros(len(X_t) + 1)
        V_t = np.zeros_like(X_t)
        q_t = np.zeros_like(X_t)
        E_t = np.zeros_like(X_t)
        theta_t = np.zeros_like(X_t)
        N_t[0] = init_conditions

        # get model variables for each period
        for t in range(len(X_t)):
            E_t[t] = self._get_E(N_t[t], X_t[t])
            q_t[t] = self._get_q(E_t[t])
            theta_t[t] = self._get_theta(q_t[t])
            V_t[t] = self._get_V(theta_t[t], N_t[t])
            N_t[t + 1] = self._get_N_tplusone(N_t[t], q_t[t], V_t[t])

        return N_t, V_t, q_t, theta_t


    def _get_mean_simu_moments(self):
        """Gets moments from each simulation, returns moment averages across simulations

        Returns:
            numpy array of moments averaged across many time series
        """
        moment_sim_list = []
        for ix in range(self.X_t.shape[1]):
            moment_sim = self._get_simu_moments_once(self.X_t[:, ix], self.U_t[:, ix], self.V_t[:, ix])
            moment_sim_list.append(moment_sim)

        # Compute mean of moments over all simulations
        mean_moments = np.array(moment_sim_list).mean(axis=0)
        return mean_moments

    def _get_simu_moments_once(self, X_t, U_t, V_t):
        """Computes moments from several time series

        Args:
            X_t (numpy array): productivity time series
            U_t (numpy array): unemployment time series
            V_t (numpy array): vacancy time series

        Returns:
            numpy array of moments (means, std, auto, corr coefs)
        """
        # get quarterly averages
        X_q = utils._get_quarterly_avg(X_t)
        U_q = utils._get_quarterly_avg(U_t)
        V_q = utils._get_quarterly_avg(V_t)
        theta_q = V_q/U_q

        # get hp-filtered series
        X_hp = utils.get_detrended_simulation(X_q)
        U_hp = utils.get_detrended_simulation(U_q)
        V_hp = utils.get_detrended_simulation(V_q)
        theta_hp = utils.get_detrended_simulation(theta_q)

        # get difference between log-series and log-hptrend
        X_d = utils.get_log_hpdif_simulation(X_q)
        U_d = utils.get_log_hpdif_simulation(U_q)
        V_d = utils.get_log_hpdif_simulation(V_q)
        theta_d = utils.get_log_hpdif_simulation(theta_q)

        # get first difference of log-series
        X_fd = utils.get_log_firstdif_simulation(X_q)
        U_fd = utils.get_log_firstdif_simulation(U_q)
        V_fd = utils.get_log_firstdif_simulation(V_q)
        theta_fd = utils.get_log_firstdif_simulation(theta_q)

        # use pct-change HP series if log returns NaNs
        if np.isnan(np.std(V_d)):
            theta_d = theta_hp
            V_d = V_hp

        # moment table includes means,
        mean_sim = np.array([U_q, V_q, theta_q, X_q]).mean(axis=1)
        std_sim = np.array([U_d, V_d, theta_d, X_d]).std(axis=1, ddof=1)
        autocorrealtion_sim = np.array([utils.AutoCorr(series) for series in [U_fd, V_fd, theta_fd, X_fd]], ndmin=2).T
        corrcoef_sim = np.corrcoef(np.array([U_d, V_d, theta_d, X_d]))
        return np.vstack((mean_sim, std_sim, autocorrealtion_sim, corrcoef_sim))


    def _solve_model(self):
        """
        """
        solver = dmp_solver()
        solver.solve_model()
        self.aE = solver.aE

if __name__ == "__main__":

    # initialize class and simulate. Note, this uses "initial guess" policy function
    simulator = dmp_simulator()
    simulator.simulate_model(continuous=False)
    moments = simulator._get_mean_simu_moments()
    print(moments)

"""dmp_irf.py

This file contains the dmp_irf class, which inherits the parameters/methods
from the dmp_model class and adds additional methods for simulating impulse 
reponse functions.

Example:

    >>> T_irf = 140
    >>> irf_nsim = 500

    >>> irf_maker = dmp_irf()
    >>> irf_maker.get_irf(T_irf, irf_nsim)
    >>> irf_maker.plot_irfs('figures/irf_{}.pdf')

"""
import utils 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dmp_model import dmp_model
from dmp_simulator import dmp_simulator
from collections import defaultdict
import matplotlib as mpl
sns.set_style("white")
mpl.rc('font', family='serif')


class dmp_irf(dmp_simulator):

    def get_irf(self, T, nsim, n_start=.9, x_pct=50, k_pct=50, shock_size=1, kshock_true=True, xshock_true=True):
        """Computes an average impulse response function from nsim simulations.

        Args:
            T (int): length of irf
            nsim (int): number of simulations

        Returns:
            numpy arrays w/ negative IRFs for X and U
        """        
        self.sim = utils.simulate_AR1(continuous=False)
        self.sim.T = T
        self.sim.parnsim = nsim        
        self.sim.get_shocks()
        self.shock_size = shock_size

        eps_n = self.sim.eps_x.copy()        
        eps = self.sim.eps_x.copy()        

        # get starting values
        _, state_index_full = self.sim.simulate_discrete(self.P, self.X_grid)
        X_full = self.X_grid[state_index_full]
        K_full = self.K_grid[state_index_full]

        X_hist = np.array([(x == X_full).sum() for x in self.X])/len(X_full.flatten())
        K_hist = np.array([(k == K_full).sum() for k in self.K])/len(K_full.flatten())    
    
        X_starts = np.percentile(X_full, x_pct)
        K_starts = np.percentile(K_full, k_pct)        

        X_n_value = X_starts - X_full.std()*shock_size*xshock_true        
        X_n_ongrid = self.X[(np.abs(self.X - X_n_value)).argmin()]
        X_ongrid = self.X[(np.abs(self.X - X_starts)).argmin()]

        K_n_value = K_starts + K_full.std()*shock_size*kshock_true
        K_n_ongrid = self.K[(np.abs(self.K - K_n_value)).argmin()]
        K_ongrid = self.K[(np.abs(self.K - K_starts)).argmin()]

        self.state_indices_n = np.argwhere((K_n_ongrid == self.K_grid) & (X_n_ongrid == self.X_grid))[0][0] 
        self.state_indices = np.argwhere((K_ongrid == self.K_grid) & (X_ongrid == self.X_grid))[0][0] 

        X_irf_n, K_irf_n, U_irf_n, V_irf_n, q_irf_n, theta_irf_n, R_irf_n = self._simulate_irf(eps_n, n_start, self.state_indices_n)
        X_irf, K_irf, U_irf, V_irf, q_irf, theta_irf, R_irf = self._simulate_irf(eps, n_start, self.state_indices)
        
        
        X_irf_n = np.nanmean(X_irf_n, axis=0) - np.nanmean(X_irf, axis=0)
        K_irf_n = np.nanmean(K_irf_n, axis=0) - np.nanmean(K_irf, axis=0)
        U_irf_n = np.nanmean(U_irf_n, axis=0) - np.nanmean(U_irf, axis=0)
        V_irf_n = (np.nanmean(V_irf_n, axis=0) - np.nanmean(V_irf, axis=0))/np.nanmean(V_irf, axis=0)
        q_irf_n = (np.nanmean(q_irf_n, axis=0) - np.nanmean(q_irf, axis=0))/np.nanmean(q_irf, axis=0)
        theta_irf_n = (np.nanmean(theta_irf_n,axis=0) - np.nanmean(theta_irf, axis=0))/np.nanmean(theta_irf,axis=0)
        R_irf_n = (np.nanmean(R_irf_n,axis=0) - np.nanmean(R_irf, axis=0))/np.nanmean(R_irf,axis=0)

        irf_dict = {"Productivity": (X_irf_n, 'Absolute'),
                    "Credit Cost": (K_irf_n, 'Absolute'),
                    "Unemployment": (U_irf_n, 'Absolute'),
                    "Vacancies": (V_irf_n, 'Percent'),
                    "Vacancy Fill Rate": (q_irf_n, 'Percent'),
                    "Labor Market Tightness": (theta_irf_n, 'Percent'),
                    "Credit Spread": (R_irf_n, 'Percent')
                    }

        return irf_dict


    def _simulate_irf(self, eps, init_conditions, initial_states):
        """Computes nsim simulated IRFs

        Args:
            sim (dmp_simulator object)
        """        
        U_irf_list = []
        V_irf_list = []
        q_irf_list = []
        theta_irf_list = []
        R_irf_list = []
        X_irf_list = []
        K_irf_list = []

        self.sim.eps_x = eps
        _, state_index_full = self.sim.simulate_discrete(self.P, self.X_grid, initial_states)
        X_full = self.X_grid[state_index_full]
        K_full = self.K_grid[state_index_full]

        for pn in range(self.sim.parnsim):
            X_t = X_full[:, pn]
            K_t = K_full[:, pn]
            N_t, V_t, q_t, theta_t, R_t = self._simulate_series(state_index_full[:, pn], init_conditions=init_conditions)
        
            # save results across loops
            X_irf_list.append(X_t)
            K_irf_list.append(K_t)
            U_irf_list.append(1 - N_t)
            V_irf_list.append(V_t)
            q_irf_list.append(q_t)
            theta_irf_list.append(theta_t)
            R_irf_list.append(R_t)

        X_irf = np.array(X_irf_list)
        K_irf = np.array(K_irf_list)
        U_irf = np.array(U_irf_list)
        V_irf = np.array(V_irf_list)
        q_irf = np.array(q_irf_list)
        theta_irf = np.array(theta_irf_list)
        R_irf = np.array(R_irf_list)

        return X_irf, K_irf, U_irf, V_irf, q_irf, theta_irf, R_irf


    def plot_irfs(self, irf_dict_list, filename_stub='figures/irf_test_{}.pdf'):
        """This function plots the IRFs. 

        Args: 

        """        
        shock_name='Productivity'
        
        
        plot_dict = defaultdict(list)
        for irf_dict in irf_dict_list:
            for key, value in irf_dict.items():
                plot_dict[key].append(value)

        
        for (irf_key, irf_vals) in plot_dict.items():

            irf_list = []
            for irf_data in irf_vals:
                irf_list.append(irf_data[0])

            fig, ax = plt.subplots(1,1, figsize=(6, 5))
            ax.plot(np.array(irf_list).T)

            # title = '{} Change from {} Standard Deviation Shock to {}'.format(change_str, self.shock_size, shock_name)
            ax.set_xlabel('Months', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=14)

            ax.set_ylabel(irf_key, fontsize=15)
            ax.legend(['Productivity', 'Credit Search', 'Both'], fontsize=14)

            # ax.legend(['Pos shock', 'Neg shock'], ncol=2)

            if filename_stub is None:
                plt.show()
            else:
                filename = filename_stub.format(irf_key)
                fig.savefig(filename, bbox_inches='tight', dpi=300)



if __name__ == "__main__":

    import statsmodels.api as sm
    from dmp_solver import dmp_solver    

    # solver = dmp_solver()
    # solver._get_initial_conditions()
    # solver.solve_model()

    irf_maker = dmp_irf()
    irf_maker._solve_model()
    # irf_maker.aE = solver.aE
    
    for n, x, k in zip([.95], [50], [50]):
        n_start = n
        n_string = str(n_start).split('.')[1]
        n_tag = 'nstart{}.pdf'.format(n_string)

        x_pct = x
        k_pct = k
        x_tag = 'Xpct{}_'.format(x_pct)
        k_tag = 'Kpct{}_'.format(k_pct)

        x_only_irf = irf_maker.get_irf(150, 100, n_start=n_start, x_pct=x_pct, k_pct=k_pct, xshock_true=True, kshock_true=False)
        k_only_irf = irf_maker.get_irf(150, 100, n_start=n_start, x_pct=x_pct, k_pct=k_pct, xshock_true=False, kshock_true=True)
        xk_only_irf = irf_maker.get_irf(150, 100, n_start=n_start, x_pct=x_pct, k_pct=k_pct, xshock_true=True, kshock_true=True)

        irf_maker.plot_irfs([x_only_irf, k_only_irf, xk_only_irf], filename_stub='figures/irf4{}_' + x_tag + k_tag + n_tag)

    
        
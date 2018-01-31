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


class dmp_irf(dmp_simulator):

    def get_irf(self, T, nsim, shock_size=1):
        """Computes an average impulse response function from nsim simulations.

        Args:
            T (int): length of irf
            nsim (int): number of simulations

        Returns:
            numpy arrays w/ negative IRFs for X and U
        """
        self._solve_model()
        self.sim = utils.simulate_AR1(continuous=True)
        self.sim.T = T
        self.sim.parnsim = nsim
        self.sim.get_shocks()
        self.shock_size = shock_size

        eps_n = self.sim.eps_x.copy()
        eps_n[0, :] -= self.shock_size

        eps = self.sim.eps_x.copy()
        
        eps_p = self.sim.eps_x.copy()
        eps_p[0, :] += self.shock_size


        X_irf_n, U_irf_n, V_irf_n, q_irf_n, theta_irf_n = self._simulate_irf(eps_n, .9)        
        X_irf, U_irf, V_irf, q_irf, theta_irf = self._simulate_irf(eps, .9)
        X_irf_p, U_irf_p, V_irf_p, q_irf_p, theta_irf_p = self._simulate_irf(eps_p, .9) 


        X_irf_n = np.exp(X_irf_n) - np.exp(X_irf)
        U_irf_n = U_irf_n - U_irf
        V_irf_n = (V_irf_n - V_irf)/V_irf
        q_irf_n = (q_irf_n - q_irf)/q_irf
        theta_irf_n = (theta_irf_n - theta_irf)/theta_irf

        self.irf_dict_n = {"Productivity": (X_irf_n, 'Absolute'),
                           "Unemployment": (U_irf_n, 'Absolute'),
                           "Vacancies": (V_irf_n, 'Percent'),
                           "Vacancy Fill Rate": (q_irf_n, 'Percent'),
                           "Labor Market Tightness": (theta_irf_n, 'Percent')
                           }

        X_irf_p = np.exp(X_irf_p) - np.exp(X_irf)
        U_irf_p = U_irf_p - U_irf
        V_irf_p = (V_irf_p - V_irf)/V_irf
        q_irf_p = (q_irf_p - q_irf)/q_irf
        theta_irf_p = (theta_irf_p - theta_irf)/theta_irf

        self.irf_dict_p = {"Productivity": (X_irf_p, 'Absolute'),
                           "Unemployment": (U_irf_p, 'Absolute'),
                           "Vacancies": (V_irf_p, 'Percent'),
                           "Vacancy Fill Rate": (q_irf_p, 'Percent'),
                           "Labor Market Tightness": (theta_irf_p, 'Percent')
                           }        


    def _simulate_irf(self, eps, init_conditions):
        """Computes nsim simulated IRFs

        Args:
            sim (dmp_simulator object)
        """        
        U_irf_list = []
        V_irf_list = []
        q_irf_list = []
        theta_irf_list = []
        X_irf_list = []

        self.sim.eps_x = eps        
        X_full = np.exp(self.sim.simulate_continuous(self.rhox, self.x_bar, self.stdx))

        for pn in range(self.sim.parnsim):
            X_t = X_full[:, pn]
            N_t, V_t, q_t, theta_t = self._simulate_series(X_t, init_conditions=init_conditions)
        
            # save results across loops
            X_irf_list.append(X_t)
            U_irf_list.append(1 - N_t)
            V_irf_list.append(V_t)
            q_irf_list.append(q_t)
            theta_irf_list.append(theta_t)

        X_irf = np.array(X_irf_list)
        U_irf = np.array(U_irf_list)
        V_irf = np.array(V_irf_list)
        q_irf = np.array(q_irf_list)
        theta_irf = np.array(theta_irf_list)

        return X_irf, U_irf, V_irf, q_irf, theta_irf 


    def plot_irfs(self, filename_stub='figures/irf_{}.pdf'):
        """This function plots the IRFs. 

        Args: 

        """        
        shock_name='Productivity'
        for (key_p, val_p), (key_n, val_n)  in zip(self.irf_dict_p.items(), self.irf_dict_n.items()):

            irf_p = val_p[0]
            irf_n = val_n[0]
            change_str = val_p[1]        

            fig, ax = plt.subplots(1,1, figsize=(6, 5))
            sns.tsplot(irf_p, color='blue', linestyle='dashed', ax=ax)
            sns.tsplot(irf_n, color='red', linestyle='solid', ax=ax)

            title = '{} Change from {} Standard Deviation Shock to {}'.format(change_str, self.shock_size, shock_name)
            ax.set_xlabel('Months')
            ax.set_ylabel(key_p)
            ax.set_title(title)
            ax.legend(['Pos shock', 'Neg shock'], ncol=2)

            if filename_stub is None:
                plt.show()
            else:
                filename = filename_stub.format(key_p)
                fig.savefig(filename)



if __name__ == "__main__":

    import statsmodels.api as sm
    from dmp_solver import dmp_solver    

    solver = dmp_solver() 
    solver._get_initial_conditions()
    solver._get_splines()
    solver.solve_model()

    irf_maker = dmp_irf()
    irf_maker.aE = solver.aE
    
    irf_maker.get_irf(140, 500)


    
        
"""credit_dmp_main.py

This file calls in the various classes required to solve, simulate, and compute IRFs 
for the DMP model specified in credit_dmp_model.py. 
"""
import time
import sys
from credit_dmp_model import credit_dmp_model
from credit_dmp_simulator import credit_dmp_simulator
from credit_dmp_solver import credit_dmp_solver
from credit_dmp_irf import credit_dmp_irf
import utils
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pickle

sns.set_style("white")
mpl.rc('font', family='serif')


# simulated moments
simulator = credit_dmp_simulator()
simulator.simulate_model(continuous=False)
moments = simulator._get_mean_simu_moments()
print('simulated moments')
print('means: ', moments[0, :])
print('st dev: ', moments[1, :])
print('skew: ', moments[2, :])
print('kurt: ', moments[3, :])
print('corr: ', moments[4:, :])
print('')
print(time.ctime())


# make some irfs (this can take a while)
irf_maker = credit_dmp_irf()
irf_maker.aE = simulator.aE    

# irfs for different initial conditions
for n, x, k in zip([.9440, .9525], [50, 90], [50, 45]):
    n_start = n
    n_string = str(n_start).split('.')[1]
    n_tag = 'nstart{}'.format(n_string)

    x_pct = x
    k_pct = k
    x_tag = 'Xpct{}_'.format(x_pct)
    k_tag = 'Kpct{}_'.format(k_pct)

    x_only_irf = irf_maker.get_irf(160, 1200, n_start=n_start, x_pct=x_pct, k_pct=k_pct, xshock_true=True, kshock_true=False)
    k_only_irf = irf_maker.get_irf(160, 1200, n_start=n_start, x_pct=x_pct, k_pct=k_pct, xshock_true=False, kshock_true=True)
    xk_only_irf = irf_maker.get_irf(160, 1200, n_start=n_start, x_pct=x_pct, k_pct=k_pct, xshock_true=True, kshock_true=True)

    # save pickle
    with open('irf' + x_tag + k_tag + n_tag + '.p', 'wb') as f:
        pickle.dump([x_only_irf, k_only_irf, xk_only_irf], f)

    irf_maker.plot_irfs([x_only_irf, k_only_irf, xk_only_irf], filename_stub='figures/irf3{}_' + x_tag + k_tag + n_tag + '.pdf')
    plt.close()
    print(time.ctime())
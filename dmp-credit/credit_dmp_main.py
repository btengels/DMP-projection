"""dmp_main.py

This file calls in the various classes required to solve, simulate, and compute IRFs 
for the DMP model specified in dmp_model.py. 
"""
import time
import sys
from dmp_model import dmp_model
from dmp_simulator import dmp_simulator
from dmp_solver import dmp_solver
from dmp_irf import dmp_irf
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
sys.path.append('../../Empirics/state_dependent_regs/')    

from state_dependent_reg_solver import state_dependent_reg_solver as solver



def regression_irf_from_simulated_data(U_t, R_t, theta_t, X_t, hp_smooth=1600):
    """
    """
    # U_q = utils._get_quarterly_avg(U_t)    
    # R_q = utils._get_quarterly_avg(R_t)
    # theta_q = utils._get_q=uarterly_avg(theta_t)    
    # X_q = utils._get_quarterly_avg(X_t)

    sns.set_style("whitegrid")
    mpl.rc('font', family='serif')    

    rhs_var = ['R', 'R_rec', 'R_exp', 'U', 'theta',  'X_pct_dev', 'recession', 'expansion']	
    lhs_var = ['U']
    columns = rhs_var
    rhs_lags = 3

    # read in the data
    df = pd.DataFrame([U_t, R_t, theta_t, X_t]).T
    df.columns=['U', 'R', 'theta', 'X']
    df['date'] = range(len(df))
    df.set_index('date', inplace=True)

    # Adds cyclical components of variables 'X' and 'U'
    df = df[np.isfinite(df['U']) & np.isfinite(df['X'])]
    for var in ['U', 'X']:

        cycle, trend = sm.tsa.filters.hpfilter(df[var], hp_smooth)
        cycle_name = var + '_cycle'
        trend_name = var + '_trend'
        pctdev_name = var + '_pct_dev'

        df[cycle_name] = cycle
        df[trend_name] = trend
        df[pctdev_name] = cycle/trend*100


    # get expansion/recession dummies
    df['recession'] = (df['U'] > df['U'].quantile(.8)).astype(float)
    df['expansion'] = (df['U'] < df['U'].quantile(.2)).astype(float)


    # interaction terms
    df['R_rec'] = df['R'] * df['recession']
    df['R_exp'] = df['R'] * df['expansion']


    # cut out any variables from csv we don't want in the regression
    df = df[columns]

    # add lagged rhs variables
    for var in rhs_var:
        for lags in range(1, rhs_lags):
            df[var + 'L' + str(lags)] = df[var].shift(lags)		


    # initialize solver object and run regressions
    regs = solver(df, lhs_var, rhs_var, monthly=False)
    regs.run_regressions()
    # regs.plot_irfs(filename='model_regression_irf.pdf')

    # # print results to console
    # print('coefficients:')
    # print(regs.coefs_df[['R', 'R_rec', 'const']].T)

    # print('standard errors:')
    # print(regs.sterr_df[['R', 'R_rec', 'const']].T)

    # print('pvalues:')
    # print(regs.pval_df[['R', 'R_rec', 'const']].T)
    return regs


def make_line_plots(array, array_rec, title, ylabel, legend, ylim, filename=None):
    """
    """
    sns.set_style("white")
    mpl.rc('font', family='serif')    
    colors = sns.color_palette()

    # plot of average price
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))   

    sns.tsplot(data=array, ax=ax, color=colors[0], ci=95)
    sns.tsplot(data=array_rec, ax=ax, color=colors[2], ci=95)

    ax.tick_params(labelsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Time (periods)', fontsize=18)

    ax.plot(0, 0, color = colors[0])
    ax.plot(0, 0, color = colors[1])
    ax.plot(0, 0, color = colors[2])
    ax.legend(legend, ncol=1, loc=0, fontsize=18)
    # ax.set_ylim(ylim)

    # plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    # plt.close()


def make_plot(U_t, R_t, nsim):
    """
    """
    df = pd.DataFrame([U_t, R_t]).T
    df.columns = ['U', 'R']
    palette = sns.color_palette()

    fig, ax = plt.subplots(1, 1, figsize=(9,6), sharex=True)
    # fig.suptitle('Unemployment and Credit Market ('+year+'Q1-2013Q1)', fontsize=15)
    ax.plot(df.index, df['U'], label='Unemployment', color=palette[0])
    ax.set_ylabel('Unemployment', color=palette[0], fontsize=14)
    # ax.set_ylim(2, 14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    for tl in ax.get_yticklabels():
        tl.set_color(palette[0])
       
    # plot Fz line
    ax2 = ax.twinx()
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.plot(df.index, df['R'], label='Credit Spread', color=palette[2])
    ax2.set_ylabel('Credit Spread', color=palette[2], fontsize=14)
    for tl in ax2.get_yticklabels():
        tl.set_color(palette[2])
    
    plt.savefig('figures/unemployment_vs_spread_simulated{}.pdf'.format(nsim), bbox_inches='tight', dpi=300)


if __name__ == "__main__":


    df = pd.read_csv('../../Data/CM_LM_quarterly_data.csv')
    # df = df[~np.isnan(df.GZ)]
    U_t = df['U'].values/100
    R_t = df['BAA10YM'].values/100
    V_t = df['V'].values/100
    X_t = df['X'].values/100

    print(time.ctime())
    simulator = dmp_simulator()
    simulator.parnsim = 2
    observed_moments = simulator._get_simu_moments_once(X_t, U_t, V_t, R_t, monthly=False)
    print('observed moments')
    print(observed_moments)
    print('')    

    stop
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

    # [[ 0.06367868  0.11979775  2.49150082  0.99665918  0.04002536]
    #  [ 0.10091531  0.14156726  0.22570164  0.01530403  0.25570047]
    #  [ 0.43796772  0.11923383  0.34715096  0.20999052         nan]
    #  [ 1.         -0.75651527 -0.91075143 -0.56939127  0.79776762]
    #  [-0.75651527  1.          0.95578776  0.84136114 -0.83815001]
    #  [-0.91075143  0.95578776  1.          0.77267347 -0.87567528]
    #  [-0.56939127  0.84136114  0.77267347  1.         -0.74919153]
    #  [ 0.79776762 -0.83815001 -0.87567528 -0.74919153  1.        ]]


    # for i in range(simulator.U_t.shape[1]):
        # make_plot(simulator.U_t[:,i]*100, simulator.rspread_t[:,i], i)
        # plt.close()
    
    irf_maker = dmp_irf()
    # irf_maker._solve_model()
    irf_maker.aE = simulator.aE    
                
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
        plt.close
        print(time.ctime())
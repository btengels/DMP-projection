"""utils.py
seed
This file contains useful functions used throughout the project. 
"""
import numpy as np
import scipy as sp
import statsmodels.api as sm


class simulate_AR1(object):
    """Object for generating and saving random shocks used in simulations"""

    def __init__(self, new_rand_draw=True, continuous=False):
        """Creates simulation_params object and sets basic simulation parameters
        
        Args:
            new_rand_draw(bool): whether to draw/save new shocks; default True
            continuous(bool): whether state space is continuous or discrete; default True
        """
        self.burnin = 500  # burn in to stationary distribution
        self.parnsim = 1  # number of 'slices' for parallelization
        self.nsim = 5  # number of simulations per parnsim
        self.len_sim = 768  # 248 quarters
        self.T = self.burnin + self.nsim*self.len_sim
        self.continuous = continuous
        self.new_rand_draw = new_rand_draw
        self.get_shocks()

    def get_shocks(self):
        """Retrieves saved random shocks or generates new ones

        """
        if self.new_rand_draw:
            self._new_shocks()
        else:
            self._load_shocks()

    def _new_shocks(self):
        """Method to generate new shocks

        """
        np.random.seed(1337)
        if self.continuous:
            self.eps_x = np.random.randn(self.parnsim, self.T).T

        else:
            self.eps_x = np.random.rand(self.parnsim, self.T).T

        # save to HDFS?
        self._save_shocks()
        
    def _save_shocks(self):
        """Saves current shocks to hdf5 file

        """
        pass

    def _load_shocks(self):
        """Load shocks from hdf5 file

        """
        pass


    def simulate_discrete(self, P, x, start_val=None):
        """Simulates AR(1) process over discrete state space 'x'

        Args:
            P (numpy array): transition probability matrix
            x (numpy array): vector of states

        """
        if self.continuous is True:
            raise('Discrete state space simulations require uniform shocks')

        simu_full = np.zeros_like(self.eps_x)
        simu_index_full = np.zeros_like(self.eps_x)        
        cQx = np.cumsum(P, axis=0)

        for i_sim in range(self.parnsim):
            shock = self.eps_x[:, i_sim]

            simu_index = np.zeros(self.T, dtype=int)         
            if start_val is not None:
                simu_index[0] = start_val
            else:
                simu_index[0] = int(len(x)/2)


            for t in range(1, self.T):    
                simu_index[t] = int((shock[t] > cQx[:, int(simu_index[t - 1])]).sum())

            sx = x[simu_index]        
            simu_index_full[:, i_sim] = simu_index
            simu_full[:, i_sim] = sx

        return simu_full, simu_index_full.astype(int)

    def simulate_continuous(self, rho, mu, std):
        """Simulates AR(1) process over continuous state space

        Args:
            rho (float): persistence of shock process, should be in [0, 1)
            mu (float): mean of AR(1) process
            std (float): standard deviation of underlying shocks

        Returns:
            np.array of dimensions (T, parnsim), each column a separate AR(1)
        """
        if self.continuous is False:
            raise('Continuous state simulation require standard normal shocks')

        simu_full = np.zeros_like(self.eps_x)
        simu_index_full = np.zeros_like(self.eps_x)   
        for i_sim in range(self.eps_x.shape[1]):
            shock = self.eps_x[:, i_sim]*std
            sx = np.zeros_like(shock)
            sx[0] = mu + shock[0]

            for t in range(1, self.T):
                sx[t] = rho*sx[t - 1] + mu*(1 - rho) + shock[t]

            simu_full[:, i_sim] = sx

        return simu_full


def get_detrended_simulation(series_t, hp_smooth=1600):
    """Given a time series, computes pct-deviation from long-term trend using an hp-filter

    Args:
        series_t (numpy array): time series
    
    Returns:
        numpy array of "filtered" time series, given by pct-deviation from hp-filter trend
    """ 
    series_pd = (series_t - series_t.mean())/series_t.mean()
    cycle, trend = sm.tsa.filters.hpfilter(series_pd, lamb=hp_smooth)
    return cycle


def get_log_hpdif_simulation(series_t, hp_smooth=1600):
    """Given a time series, computes deviation of logged series from long-term trend using an hp-filter

    Args:
        series_t (numpy array): time series
    
    Returns:
        numpy array of "filtered" logged time series, given by deviation from hp-filter trend
    """ 
    log_series = np.log(series_t)
    cycle, trend = sm.tsa.filters.hpfilter(log_series, lamb=hp_smooth)
    return cycle


def get_log_firstdif_simulation(series_t):
    """Computes logged first-difference of a time series

    Args:
        series_t (numpy array): time series

    Returns:
        numpy array of logged-first-differenced series
    """
    log_series = np.log(series_t)
    return log_series[1:] - log_series[:-1]

def _get_quarterly_avg(series_t):
    """For a given monthly time-series, compute series of quarterly averages
    
    Args: 
        series_t (numpy array): monthly time series
    """
    len_q = int(len(series_t)/3)
    series_q = series_t.reshape([len_q, 3]).mean(axis=1)
    return series_q    


def rouwen(rho, mu, step_size, n_states):
    """
    Adapted from Lu Zhang and Karen Kopecky. Construct transition probability
    matrix for discretizing an AR(1) process. See Rouwenhorst (1995).

    Args:
	    rho(float): persistence (close to one)
	    mu(float): mean and the middle point of the discrete state space
	    step_size(float): step size of the even-spaced grid
	    n_states(int): n_statesber of grid points on the discretized process

    Returns:
    	np.array - discrete state space
    	np.array - transition probability matrix
    """

    # discrete state space
    dscSp = np.linspace(mu -(n_states-1)/2*step_size, mu +(n_states-1)/2*step_size, n_states).T

    # transition probability matrix
    q = p = (rho + 1)/2.
    transP = np.array([[p**2, p*(1-q), (1-q)**2],
    				   [2*p*(1-p), p*q+(1-p)*(1-q), 2*q*(1-q)],
    				   [(1-p)**2, (1-p)*q, q**2]]).T

    while transP.shape[0] <= n_states - 1:
        len_P = transP.shape[0]
        transP = p*np.vstack((np.hstack((transP,np.zeros((len_P,1)))), np.zeros((1, len_P+1)))) \
        + (1-p)*np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
        + (1-q)*np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
        + q*np.vstack((np.zeros((1, len_P+1)), np.hstack((np.zeros((len_P, 1)), transP))))

        transP[1:-1] /= 2.

    # ensure columns sum to 1
    if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
        print('Problem in rouwen routine!')
        return None
    else:
        return transP.T, dscSp


def AutoCorr(X, lags=[1]):
	"""Evaluate autocorrelations of vectors X of orders defined in lags

	Args:
		X (np.array): a time series of observations
	  	lags (list): a list of positive integers defining the orders of autocorrelation 

	Returns:
		np.array of autocorrelations of orders in lags
	"""
	corr_list = []
	lags.sort()

	for j in range(len(lags)):
	    tmp = np.corrcoef(X[int(lags[j]):], X[: int(-lags[j])])
	    corr_list.append(tmp[0, 1])

	return np.array(corr_list)


if __name__ == '__main__':

    rho = .95**(1/3)
    mu = 0
    std = .00625
    n_states = 15
    step_size = 2*std/np.sqrt((1 - rho**2)*(n_states-1))
    
    P, x = rouwen(rho, mu, step_size, n_states)
    sim = simulate_ar1()
    Xt, foo = sim.simulate_discrete(P, x)
    Xt = np.exp(Xt)    
    for i in range(4):
        X = _get_quarterly_avg(Xt[0:3*5000, i])
        print('discrete:', AutoCorr(X), X.std())

    # X = np.sin(np.linspace(0, 10, 420)) + np.sin(np.linspace(0, 70, 420))*.25 + 2       
        X_hp = get_detrended_simulation(X)

        X_d = get_log_hpdif_simulation(X)
    
        X_fd = get_log_firstdif_simulation(X)
        print(AutoCorr(X_fd))


    # sim = simulate_ar1(continuous=True)
    # X = sim.simulate_continuous(rho, mu, std)    
    # print('continnuous:', AutoCorr(X[:, 0]), X[:,0].std())

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

irf_list = []
for irf_data in ['irfXpct50_Kpct50_nstart944.p', 'irfXpct90_Kpct45_nstart9525.p', 'irfXpct5_Kpct95_nstart9056.p' ]:
	with open(irf_data, 'rb') as f:
		[x_only_irf, k_only_irf, xk_only_irf] = pickle.load(f)
		irf_list.append(k_only_irf['Unemployment'][0])
	
        
fig, ax = plt.subplots(1,1, figsize=(6, 5))
ax.plot(np.array(irf_list).T)

# title = '{} Change from {} Standard Deviation Shock to {}'.format(change_str, self.shock_size, shock_name)
ax.set_xlabel('Months', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.set_ylabel('Percent', fontsize=15)
ax.legend(['Productivity', 'Credit Search', 'Both'], fontsize=14)

# ax.legend(['Pos shock', 'Neg shock'], ncol=2)

# if filename_stub is None:
plt.show()

    # filename = 'test_irf.pdf'
    # fig.savefig(filename, bbox_inches='tight', dpi=300)	
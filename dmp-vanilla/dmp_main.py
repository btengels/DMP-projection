"""dmp_main.py

This file calls in the various classes required to solve, simulate, and compute IRFs 
for the DMP model specified in dmp_model.py. 
"""
from dmp_model import dmp_model
from dmp_simulator import dmp_simulator
from dmp_solver import dmp_solver
from dmp_irf import dmp_irf


# solve the model
solver = dmp_solver() 
solver.solve_model()

# simulat the model (discrete)
simulator = dmp_simulator()
simulator.simulate_model(continuous=False)
moments = simulator._get_mean_simu_moments()
print(moments)

# [[ 0.06312557  0.05263913  0.84075271  0.99910579]
#  [ 0.02437562  0.02719983  0.04796507  0.0130017 ]
#  [ 0.37386197 -0.12763594  0.18771553  0.18770723]
#  [ 1.         -0.72717051 -0.92056497 -0.91430592]
#  [-0.72717051  1.          0.93735569  0.93723285]
#  [-0.92056497  0.93735569  1.          0.99672546]
#  [-0.91430592  0.93723285  0.99672546  1.        ]]


# simulate the model (continuous)
simulator.simulate_model(continuous=True)
moments = simulator._get_mean_simu_moments()
print(moments)

# [[ 0.06289569  0.05278485  0.84556396  1.00067105]
#  [ 0.02356313  0.02661809  0.04652098  0.01271316]
#  [ 0.33261598 -0.13437431  0.18044341  0.18078724]
#  [ 1.         -0.71741831 -0.91681767 -0.91125926]
#  [-0.71741831  1.          0.93575016  0.93534647]
#  [-0.91681767  0.93575016  1.          0.99693663]
#  [-0.91125926  0.93534647  0.99693663  1.        ]]

# compute/plot IRFs
irf_maker = dmp_irf()
irf_maker.get_irf(140, 500)
irf_maker.plot_irfs('figures/irf_{}.pdf')
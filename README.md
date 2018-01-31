# DMP-projection
Code for accurately solving various kinds of Diamond-Mortenson-Pissarides models for labor-market search

## Setting up your python environment
Our code does not require many libraries beyond numpy, scipy, and statsmodels. The full list of required libraries and their versions is found in `requirements.txt`. To create a python environment with these libraries, execute the following after cloning the repo:

    conda create -n dmp_projection python=3.6
    source activate dmp_projection
    pip install -r requirements.txt

On OSX, sometimes matplotlib has problems in virtual environments. If this problem arises remove `matplotlib` from the requirements.txt and replace the first line of the previous snippet with this:

    conda create -n dmp_projection python=3.6 matplotlib

If you want to execute the code in an interactive setting, you'll also want to `pip install jupyter` which will give you `ipython`, `jupyter notebooks` and everything else you could want to run the code interactively. 


## Model files
The files for building/solving a model are located in the `main` folder. 

  - `dmp_model.py` : contains model parameters and the most fundmental model equations
  - `dmp_solver.py`: contains the models residual function and a routine for solving it
  - `dmp_simulator.py`: contains routines for simulating the model, computing moments if desired
  - `dmp_irf.py`: contains functions for simulating/plotting impulse response functions
  - `utils.py`: contains generic functions not related to the model


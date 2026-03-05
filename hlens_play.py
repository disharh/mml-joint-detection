import sys
# Plotting

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
plt.rc('image', interpolation='none')

# Basic imports
import os
import numpy as np
from copy import deepcopy
import time
import corner
from pprint import pprint
import pickle as pkl

# JAX
import jax
jax.config.update("jax_enable_x64", True)  # comment for single precision
import jax.numpy as jnp

# optimizers
import optax

# NUTS Hamiltonian MC sampling
import blackjax

# probabilistic model and variational inference
import numpyro
import numpyro.distributions as dist
from numpyro import infer
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

# Herculens
import herculens as hcl
from herculens.Util import param_util, plot_util


print("done")

from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory
# BayBE kernels (NOT GPyTorch!)
# BE CAREFUL !!
# Kernels share names in gpytorch and baybe BUT are not the same!!
from baybe.kernels import ScaleKernel, MaternKernel, RBFKernel
from baybe.priors.basic import GammaPrior, LogNormalPrior
# from gpytorch.priors import UniformPrior
import math
import numpy as np


class AdaptiveKernelFactory(KernelFactory):
    """
    Normalize each parameter group to have similar influence.
    Simpler than full block ARD, but effective.
    """

    def __init__(self, n_dim=None, kernel_name_user='Matern'):
        self.n_dim = None
        self.kernel_name_user = kernel_name_user
    
    def __call__(self, searchspace, train_x, train_y):

        if self.n_dim is None:
            self.n_dim = len(searchspace.comp_rep_columns)
        
        x = math.sqrt(self.n_dim)
        l_mean = 0.4 * x + 4.0 # decided by fitting the result points.
        
        lengthscale_prior = GammaPrior(2.0*l_mean, 2.0)
        lengthscale_initial_value = l_mean
        outputscale_prior = GammaPrior(1.0*l_mean, 1.0) # can use a smaller rate for larger variance.
        outputscale_initial_value = l_mean


        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
            ) if self.kernel_name_user in ['Matern', 'matern'] else RBFKernel(lengthscale_prior=lengthscale_prior, lengthscale_initial_value=lengthscale_initial_value),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
        )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import Tensor
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

class GPBase(gpytorch.models.ExactGP):
    """
    The base class for any Gaussian process model.

    Args:
        train_x: The training features.
        train_y: The training targets.
        likelihood: The Gaussian likelihood that defines the observational
                    distribution.

    Returns:
        Calling this model will return the posterior of the latent Gaussian
        process when conditioned on the training data. The output will be a
        obj gpytorch.distributions.MultivariateNormal.

    """

    def __init__(self, train_x: Tensor, train_y: Tensor, 
                 likelihood: GaussianLikelihood) -> Tensor:
        """
        Initialise a Gaussian process model.

        Args:
            train_x: The training features.
            train_y: The training targets.
            likelihood: The Gaussian likelihood that defines the observational
                        distribution.

        Returns:
            None.

        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        ard_dims = train_x.shape[1]
        self.covar_module = gpytorch.kernels.ScaleKernel( 
                            gpytorch.kernels.MaternKernel(ard_num_dims=ard_dims) )
        
        #self.covar_module = gpytorch.kernels.ScaleKernel(
        #    gpytorch.kernels.RBFKernel()
        #)
        

    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Forward input data to a Gaussian process model.

        Args:
            x: The input data.

        Returns:
            The posterior of the latent Gaussian process when conditioned
            on the training data.

        """
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)
    
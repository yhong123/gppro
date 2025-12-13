#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 14:50:45 2025

@author: localadmin
"""

import gpytorch

class GPBase(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        ard_dims = train_x.shape[1]
        self.covar_module = gpytorch.kernels.ScaleKernel( gpytorch.kernels.MaternKernel(ard_num_dims=ard_dims) )
        '''
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        '''

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)
    
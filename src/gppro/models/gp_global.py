#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:14:25 2023.

@author: yhong.
"""

import torch
from torch import Tensor
import gpytorch
from .gp_base import GPBase


class GPGlobal():
    """ A single global Gaussian process model using GPBase. """
    
    def __init__(self) -> None:
        """ Initialise a product-of-experts Gaussian process model. """
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        
    def train(self, train_x: Tensor, train_y: Tensor) -> None:
        """
        Fit a single global Gaussian process model using the training data.

        Args:
            train_x: The training features.
            train_y: The training targets.

        Returns:
            None.

        """
        x_train_torch = train_x 
        y_train_torch = train_y 
        
        x_train_torch = x_train_torch.contiguous().to(dtype=torch.float32) 
        y_train_torch = y_train_torch.contiguous().to(dtype=torch.float32)
        
        self.gp = GPBase(x_train_torch, y_train_torch, self.likelihood)
        
        self.gp.train() 
        self.likelihood.train()

        # Use Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.gp.parameters()},
        ], lr=0.1)

        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, 
                                                        self.gp)
        
        # Train the GP
        for i in range(50):
            optimizer.zero_grad()
            output1 = self.gp(x_train_torch)
            loss = -mll1(output1, y_train_torch) 
            loss.backward()
            print(f"[{i+1}/50] Loss: {loss.item():.4f}")
            optimizer.step()    
            
        self.gp.eval() 
        self.likelihood.eval()
        
        
    def predict(self, test_x: Tensor) -> Tensor:
        """
        Predict the posterior target values for the input data.

        Args:
            test_x: The input data.

        Returns:
            The posterior mean and variance of the targets.

        """
        x_test_torch = test_x 
        x_test_torch = x_test_torch.contiguous().to(dtype=torch.float32)  
        
        self.gp.eval() 
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(x_test_torch))
            
        return pred.mean, pred.variance
            
        
        
        
        
        
        
        
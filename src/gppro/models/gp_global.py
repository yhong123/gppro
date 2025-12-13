#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 18:05:02 2025

@author: localadmin
"""

import torch
import gpytorch
from .gp_base import GPBase


class GPGlobal():
    def __init__(self):
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        
    def train(self, train_x, train_y):
        
        X_train_torch = train_x #torch.from_numpy(train_x) #torch.tensor(X_train)
        y_train_torch = train_y #torch.from_numpy(train_y)
        
        X_train_torch = X_train_torch.contiguous().to(dtype=torch.float32)   #double()
        y_train_torch = y_train_torch.contiguous().to(dtype=torch.float32)
        
        self.gp = GPBase(X_train_torch, y_train_torch, self.likelihood)
        
        self.gp.train(); self.likelihood.train()
        #gp2.train(); likelihood2.train()

        # Use Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.gp.parameters()},
            #{'params': gp2.parameters()}
        ], lr=0.1)

        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        #mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, gp2)

        # Train both GPs
        for i in range(50):
            optimizer.zero_grad()
            output1 = self.gp(X_train_torch)
            #output2 = gp2(X2)
            loss = -mll1(output1, y_train_torch) #- mll2(output2, Y2)
            loss.backward()
            print(f"[{i+1}/50] Loss: {loss.item():.4f}")
            optimizer.step()    
            
        self.gp.eval(); self.likelihood.eval()
        
        
    def predict(self, test_x):
        
        X_test_torch = test_x #torch.from_numpy(test_x) #torch.tensor(X_train)
        #y_test_torch = torch.from_numpy(test_y)
        
        X_test_torch = X_test_torch.contiguous().to(dtype=torch.float32)   #double()
        #y_test_torch = y_test_torch.contiguous().to(dtype=torch.float32)
        
        self.gp.eval(); self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(X_test_torch))
            
        return pred.mean, pred.variance
            
        
        
        
        
        
        
        
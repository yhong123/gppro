#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 17:02:37 2025

@author: localadmin
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error #, log_loss, r2_score


def compute_evaluation(y_true, y_pred):
    """Compute Mean Absolute Error & Root Mean Squared Error."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #nll = log_loss(y_test, preds)

    print("MAE (minutes):", mae)
    print("RMSE (minutes):", rmse)
    #print("NLL (minutes):", nll)


def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""

def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""

def cross_validate(experts, k=5):
    """K-fold CV for hyperparameter tuning."""

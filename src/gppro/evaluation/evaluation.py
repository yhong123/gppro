#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 17:02:37 2025

@author: localadmin
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error #, log_loss, r2_score


def compute_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Compute Mean Absolute Error & Root Mean Squared Error."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("MAE (minutes):", mae)
    print("RMSE (minutes):", rmse)



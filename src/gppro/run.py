#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 17:22:57 2025

@author: localadmin
"""

import torch
import math
#import pandas as pd
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, r2_score
#from xgboost import XGBRegressor
#from sklearn.ensemble import HistGradientBoostingRegressor #GradientBoostingRegressor
#from datetime import datetime
#from math import radians, sin, cos, sqrt, atan2
#from utils import compute_stop_arrivals
#import os
#from data.loader import load_bus_stop_data, load_gps_data
#from features.preprocessing import compute_stop_arrivals
#from features.feature_engineering import add_temporal_features, get_target, get_features, normalise_data_minmaxscaler
from evaluation.evaluation import compute_evaluation
from models.gp_global import GPGlobal
from models.gp_pro import GPPro


# ===============================================================
#  LOAD DATA
# ===============================================================

# Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
# in parallel.
train_x1 = torch.linspace(0, 2, 501).unsqueeze(-1)
train_y1 = torch.sin(train_x1 * (2 * math.pi)).squeeze()
train_y1.add_(torch.randn_like(train_y1).mul_(0.01))
test_x1 = torch.linspace(0, 2, 501).unsqueeze(-1)
test_y1 = torch.sin(test_x1 * (2 * math.pi)).squeeze()

train_x2 = torch.linspace(0, 1, 501).unsqueeze(-1)
train_y2 = torch.sin(train_x2 * (2 * math.pi)).squeeze()
train_y2.add_(torch.randn_like(train_y2).mul_(0.01))
test_x2 = torch.linspace(0, 1, 501).unsqueeze(-1)
test_y2 = torch.sin(test_x2 * (2 * math.pi)).squeeze()

# Combined sets of data
#train_x12 = torch.cat((train_x1.unsqueeze(0), train_x2.unsqueeze(0)), dim=0).contiguous()
#train_y12 = torch.cat((train_y1.unsqueeze(0), train_y2.unsqueeze(0)), dim=0).contiguous()
#test_x12 = torch.cat((test_x1.unsqueeze(0), test_x2.unsqueeze(0)), dim=0).contiguous()


X_train = train_x2
y_train = train_y2
X_test = test_x2
y_test = test_y2

# ===============================================================
#  MODEL - GP 
# ===============================================================

print("\n Model: GP")

gp = GPGlobal()
gp.train(X_train, y_train) #.values)

test_mean, test_var = gp.predict(X_test)  # return torch.Tensor
#val_mean, val_var = gp_bus.predict(X_val)
#print("test_mean type: ", type(test_mean))
#print("y_test type: ", type(y_test))

print("Test:")
mae, rmse = compute_evaluation(y_test, test_mean)
print("MAE:", mae)
print("RMSE:", rmse)
#print("Val:")
#compute_evaluation(y_val, val_mean)
#print("", torch.mean(torch.abs(y_test - test_mean)))



# ===============================================================
#  MODEL - GP-pro
# ===============================================================

print("\n Model: GP-pro")

gp_pro = GPPro()
gp_pro.train(X_train, y_train) #.values)
test_mean, test_var = gp_pro.predict(X_test)   # return np.ndarray
#val_mean, val_var = gp_pro.predict(X_val)
#print("test_mean type: ", type(test_mean))

print("Test:")
mae, rmse = compute_evaluation(y_test, test_mean)
print("MAE:", mae)
print("RMSE:", rmse)
#print("Val:")
#compute_evaluation(y_val, val_mean)
#print("", torch.mean(torch.abs(y_test - torch.from_numpy(test_mean))))





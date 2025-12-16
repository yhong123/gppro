""" Functions for generating synthetic datasets. """

import torch
import math
from torch import Tensor

def generate_synthetic_data_1() -> Tensor:
    """
    Generate a sine dataset.

    Returns:
        A sine dataset.

    """
    train_x1 = torch.linspace(0, 2, 501).unsqueeze(-1)
    train_y1 = torch.sin(train_x1 * (2 * math.pi)).squeeze()
    train_y1.add_(torch.randn_like(train_y1).mul_(0.01))
    test_x1 = torch.linspace(0, 2, 501).unsqueeze(-1)
    test_y1 = torch.sin(test_x1 * (2 * math.pi)).squeeze()
    
    return train_x1, train_y1, test_x1, test_y1


def generate_synthetic_data_2() -> Tensor:
    """
    Generate a sine dataset.

    Returns:
        A sine dataset.

    """
    train_x2 = torch.linspace(0, 1, 501).unsqueeze(-1)
    train_y2 = torch.sin(train_x2 * (2 * math.pi)).squeeze()
    train_y2.add_(torch.randn_like(train_y2).mul_(0.01))
    test_x2 = torch.linspace(0, 1, 501).unsqueeze(-1)
    test_y2 = torch.sin(test_x2 * (2 * math.pi)).squeeze()
    
    return train_x2, train_y2, test_x2, test_y2


def generate_synthetic_data_3() -> Tensor:
    """
    Generate a sine dataset.

    Returns:
        A sine dataset.

    """
    train_x1 = torch.linspace(0, 2, 501).unsqueeze(-1)
    train_y1 = torch.sin(train_x1 * (2 * math.pi)).squeeze()
    train_y1.add_(torch.randn_like(train_y1).mul_(0.01))
    test_x1 = torch.linspace(0, 2, 501).unsqueeze(-1)
    
    train_x2 = torch.linspace(0, 1, 501).unsqueeze(-1)
    train_y2 = torch.sin(train_x2 * (2 * math.pi)).squeeze()
    train_y2.add_(torch.randn_like(train_y2).mul_(0.01))
    test_x2 = torch.linspace(0, 1, 501).unsqueeze(-1)
    
    # Combined sets of data
    train_x12 = torch.cat((train_x1.unsqueeze(0), 
                           train_x2.unsqueeze(0)), dim=0).contiguous()
    train_y12 = torch.cat((train_y1.unsqueeze(0), 
                           train_y2.unsqueeze(0)), dim=0).contiguous()
    test_x12 = torch.cat((test_x1.unsqueeze(0), 
                          test_x2.unsqueeze(0)), dim=0).contiguous()
    
    return train_x12, train_y12, test_x12




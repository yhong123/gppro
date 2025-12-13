"""An example set of tests."""

import unittest
import math
import torch
#from gppro.models import GPGlobal
#from gppro.models import GPPro
#from sklearn.metrics import mean_absolute_error

# Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
# in parallel.
train_x1 = torch.linspace(0, 2, 11).unsqueeze(-1)
train_y1 = torch.sin(train_x1 * (2 * math.pi)).squeeze()
train_y1.add_(torch.randn_like(train_y1).mul_(0.01))
test_x1 = torch.linspace(0, 2, 51).unsqueeze(-1)
test_y1 = torch.sin(test_x1 * (2 * math.pi)).squeeze()

'''
class TestGPGlobal(unittest.TestCase):
    
    def test_train_predict(self) -> None:
        """Test is merely a placeholder."""
        gp = GPGlobal()
        gp.train(train_x1, train_y1) 
    
        test_mean, test_var = gp.predict(test_x1)
        
        preds1 = test_mean
        mae = mean_absolute_error(test_y1, preds1)
        self.assertLess(mae, 0.1)
        #mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        #self.assertLess(mean_abs_error1.squeeze().item(), 0.1)
    
    
class TestGPPro(unittest.TestCase):
    
    def test_train_predict(self) -> None:
        """Test is merely a placeholder."""
        gp_pro = GPPro()
        gp_pro.train(train_x1, train_y1) 
    
        test_mean, test_var = gp_pro.predict(test_x1)
        
        preds1 = test_mean
        mae = mean_absolute_error(test_y1, preds1)
        self.assertLess(mae, 0.1)
        #mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        #self.assertLess(mean_abs_error1.squeeze().item(), 0.1)
'''

def test_stupid_example() -> None:
    """Test is merely a placeholder."""
    assert True


if __name__ == '__main__':
    unittest.main()
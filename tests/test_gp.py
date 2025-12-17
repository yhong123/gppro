"""Set of tests."""


from gppro.models import GPGlobal
from gppro.models import GPPro
from gppro.data import generate_synthetic_data_1
from sklearn.metrics import mean_absolute_error



def test_gpglobal_train_predict() -> None:
    """ Test GP global model. """
    train_x1, train_y1, test_x1, test_y1 = generate_synthetic_data_1()
    gp = GPGlobal()
    gp.train(train_x1, train_y1) 

    test_mean, test_var = gp.predict(test_x1)
    
    preds1 = test_mean
    mae = mean_absolute_error(test_y1, preds1)
    mae_max = 0.1
    assert mae < mae_max, f"MAE too large: {mae}"
    
    
def test_gppro_train_predict() -> None:
    """ Test product-of-experts GP model. """
    train_x1, train_y1, test_x1, test_y1 = generate_synthetic_data_1()
    gp_pro = GPPro()
    gp_pro.train(train_x1, train_y1) 

    test_mean, test_var = gp_pro.predict(test_x1)
    
    preds1 = test_mean
    mae = mean_absolute_error(test_y1, preds1)
    mae_max = 0.1
    assert mae < mae_max, f"MAE too large: {mae}"
    


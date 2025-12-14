"""An example set of tests."""


from gppro.models import GPGlobal
from gppro.models import GPPro
from gppro.data import generate_synthetic_data_1
from sklearn.metrics import mean_absolute_error



def test_gpglobal_train_predict() -> None:
    """Test is merely a placeholder."""
    train_x1, train_y1, test_x1, test_y1 = generate_synthetic_data_1()
    gp = GPGlobal()
    gp.train(train_x1, train_y1) 

    test_mean, test_var = gp.predict(test_x1)
    
    preds1 = test_mean
    mae = mean_absolute_error(test_y1, preds1)
    assert mae < 0.1, f"MAE too large: {mae}"
    
    
def test_gppro_train_predict() -> None:
    """Test is merely a placeholder."""
    train_x1, train_y1, test_x1, test_y1 = generate_synthetic_data_1()
    gp_pro = GPPro()
    gp_pro.train(train_x1, train_y1) 

    test_mean, test_var = gp_pro.predict(test_x1)
    
    preds1 = test_mean
    mae = mean_absolute_error(test_y1, preds1)
    assert mae < 0.1, f"MAE too large: {mae}"
    


def test_stupid_example() -> None:
    """Test is merely a placeholder."""
    assert True



"""
Run an example of using the calibrated product-of-experts Gaussian process 
model for a regression task.

"""

from data.synthetic_data import generate_synthetic_data_2
from evaluation.evaluation import compute_evaluation
from models.gp_global import GPGlobal
from models.gp_pro import GPPro


# ===============================================================
#  LOAD DATA
# ===============================================================

X_train, y_train, X_test, y_test = generate_synthetic_data_2()


# ===============================================================
#  MODEL - GP 
# ===============================================================

print("\n Model: GP") # noqa: T201

gp = GPGlobal()
gp.train(X_train, y_train) #.values)

test_mean, test_var = gp.predict(X_test)  # return torch.Tensor

print("Test:") # noqa: T201
mae, rmse = compute_evaluation(y_test, test_mean)
print("MAE:", mae) # noqa: T201
print("RMSE:", rmse) # noqa: T201


# ===============================================================
#  MODEL - GP-pro
# ===============================================================

print("\n Model: GP-pro") # noqa: T201

gp_pro = GPPro()
gp_pro.train(X_train, y_train) #.values)
test_mean, test_var = gp_pro.predict(X_test)   # return np.ndarray

print("Test:") # noqa: T201
mae, rmse = compute_evaluation(y_test, test_mean)
print("MAE:", mae) # noqa: T201
print("RMSE:", rmse) # noqa: T201





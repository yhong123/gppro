""" Functions for evaluating models. """

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error #, log_loss, r2_score


def compute_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Compute Mean Absolute Error & Root Mean Squared Error.

    Args:
        y_true: The actual target values.
        y_pred: The predicted target values.

    Returns:
        Computed MAE & RMSE.

    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse


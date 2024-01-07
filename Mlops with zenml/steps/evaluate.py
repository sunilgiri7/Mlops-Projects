import logging
import pandas as pd
from zenml import step 
from src.evaluation import MSE, R2, rmse
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated


@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
    ) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"]
    ]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = rmse()
        rmsee = rmse_class.calculate_scores(y_test, prediction)

        return r2, rmsee
    except Exception as e:
        logging.error("Failed to calculate matrices: {}".format(e))
        raise e
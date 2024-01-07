import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            logging.info('MSE: {}'.format(mse))
            return mse
        except Exception as e:
            logging.error('Unable to calculate MSE: {}'.format(e))
            raise e
        
class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating R2 scores')
            r2 = r2_score(y_true, y_pred)
            logging.info('R2 scores: {}'.format(r2))
            return r2
        except Exception as e:
            logging.error('Unable to calculate r2 score: {}'.format(e))
            raise e
        
class rmse(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE')
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info('RMSE: {}'.format(rmse))
            return rmse
        except Exception as e:
            logging.error('Unable to calculate RMSE: {}'.format(e))
            raise e
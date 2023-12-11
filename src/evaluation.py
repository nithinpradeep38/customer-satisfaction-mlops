import logging
from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

class Evaluation(ABC):

    """
    Abstract class defining the strategy for evaluating model performance
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray)-> float:
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: numpy array of true label
            y_pred: numpy array of predicted label

        Returns:
            mse: float
        """
        try:
            logging.info("Calculating MSE")
            mse= mean_squared_error(y_true, y_pred)
            logging.info("The mean squared error is: "+ str(mse))
            return mse
        except Exception as e:
            logging.error("Exception occured in calculating the metric: {}".format(e))
            raise e
        
class R2_Score(Evaluation):
    """
    Evaluation strategy that uses R-squared metric
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: numpy array of true label
            y_pred: numpy array of predicted label

        Returns:
            r2_score: float
        """
        try:
            logging.info("Calculating R2-score")
            r2= r2_score(y_true, y_pred)
            logging.info("The R-squared metric is: "+ str(r2))
            return r2
        except Exception as e:
            logging.error("Exception occured in calculating the metric: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (MSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: numpy array of true label
            y_pred: numpy array of predicted label

        Returns:
            rmse: float
        """
        try:
            logging.info("Calculating RMSE")
            rmse= np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("The root mean squared error is: "+ str(rmse))
            return rmse
        except Exception as e:
            logging.error("Exception occured in calculating the metric: {}".format(e))
            raise e
        
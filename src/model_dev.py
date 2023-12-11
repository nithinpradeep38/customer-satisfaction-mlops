import logging
from abc import ABC, abstractmethod
import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    This is the abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model on the given data 
        """
        pass

    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        """
        Optimizes the hyperparameters of the model

        trial: Optuna trial object
        """

        pass

class LinearRegressionModel(Model):

    def train(self, X_train, y_train, **kwargs):
        try:
            reg= LinearRegression(**kwargs)
            reg.fit(X= X_train, y= y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
    
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        reg= self.train(X_train,y_train)
        return reg.score(X_test,y_test)
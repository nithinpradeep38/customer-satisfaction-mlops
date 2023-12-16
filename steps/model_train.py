import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from steps.config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker= Client().active_stack.experiment_tracker #to track experiments

@step(experiment_tracker= experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
)-> RegressorMixin:
    """
    Trains the model on the ingested data

    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model= None
        if config.model_name== "LinearRegression":
            model= LinearRegressionModel()
            mlflow.sklearn.autolog()
            trained_model= model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported or listed".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
    
        


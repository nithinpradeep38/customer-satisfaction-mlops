import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2_Score, RMSE

from typing_extensions import Annotated
from typing import Tuple

import mlflow
from zenml.client import Client

experiment_tracker= Client().active_stack.experiment_tracker #to track experiments

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.Series)-> Tuple[Annotated[float,"r2_score"], Annotated[float, "rmse"]]:
    
    try:
        prediction= model.predict(X_test)
        r2_class= R2_Score()
        r2_score= r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("r2_score",r2_score)

        rmse_class= RMSE()
        rmse= rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse",rmse)

        return r2_score, rmse
    except Exception as e:
        logging.error("Error in evaluating the model: {}".format(e))
        raise e
    

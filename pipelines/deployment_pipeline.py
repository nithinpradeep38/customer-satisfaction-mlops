import numpy as np 
import pandas as pd 
from zenml import pipeline, step
from zenml.config import DockerSettings

from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.model_evaluation import evaluate_model

"""
Objective: Need to train the model and if the accuracy meets threshold, deploy the model.
"""

docker_settings= DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy >= config.min_accuracy

@pipeline(enable_cache=False, settings= {'docker': docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float= 0,
    workers: int=1,
    timeout: int= DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    #Link all the steps together
    df= ingest_data()
    X_train,X_test,y_train,y_test= clean_data(df)
    model= train_model(X_train,X_test,y_train,y_test)
    r2_score, rmse= evaluate_model(model, X_test,y_test)
    deployment_decision= deployment_trigger(accuracy=r2_score) #deploy only if the value greater than the set min_accuracy 
    
    mlflow_model_deployer_step(
        model= model,
        deploy_decision= deployment_decision,
        workers= workers,
        timeout= timeout
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline():
    pass
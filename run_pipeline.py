from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__== "__main__":
    #run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri()) #to get location of the experiment_tracker files
    training_pipeline()

    
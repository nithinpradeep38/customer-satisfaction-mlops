# customer satisfaction (mlops)
**Problem statement**: For a given customer's historical data, we are tasked to predict the review score for the next order or purchase. We will be using the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. 

## :snake: Python Requirements

Install zenml
```pip install "zenml["server"]"```

Install requirements
`pip install -r requirements.txt`

To start zenml locally

```bash
zenml init
zenml up
```
To run deployment pipeline, also install integrations using zenml. We need to register the experiment tracker and model deployer components

```
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

If you see Connectionerror while registering experiment-tracker, try disconnecting zenml by running `zenml disconnect`.
If the issue persists,try 

`export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` before running `zenml up`

### Reviewing MLFlow experiment-tracker in UI
After registering MLFlow, run the pipeline along with the following to track the backend store location

`print(Client().active_stack.experiment_tracker.get_tracking_uri())`

Run the below. The MLflow experiment tracker will be active in 5000 port. You can inspect your experiment runs within MLflow and compare different runs.

`mlflow ui --backend-store-uri "<enter backend store location>"`
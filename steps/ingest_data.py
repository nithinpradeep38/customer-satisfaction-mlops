import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self)-> None:
        """
        Initialize the data ingestion class.
        """
    def get_data(self)-> pd.DataFrame:
        logging.info(f"Ingesting data")
        df= pd.read_csv("./data/olist_customers_dataset.csv")
        return df
    
@step
def ingest_data()-> pd.DataFrame:
    try:
        ingest_data= IngestData()
        df= ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e

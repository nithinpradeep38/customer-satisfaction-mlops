import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreProcessStrategy, DataDivideStrategy,DataCleaning
from typing_extensions import Union, Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]
]:
    try:
        preprocess_strategy= DataPreProcessStrategy()
        data_cleaning= DataCleaning(data=df,strategy= preprocess_strategy)
        preprocessed_data= data_cleaning.handle_data()

        divide_strategy= DataDivideStrategy()
        data_cleaning= DataCleaning(data=preprocessed_data, strategy= divide_strategy)
        X_train,X_test,y_train,y_test= data_cleaning.handle_data()
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(e)
        raise e
    

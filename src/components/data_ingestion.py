import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTraining
from src.components.model_training import ModelTrainingConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion class and sets up the configuration.
        """
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entering the data ingestion method or component.")

        try:
            df = pd.read_csv('notebook/data/stud.csv')          ## data source can be local file, MongoDB, MySQL etc.
            logging.info("Read the dataset as a pandas DataFrame.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train-test split initiated.')
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Ingestion of data completed.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ =  data_transformation.initiate_data_transformation(train_data, test_data)

    model_training = ModelTraining()
    #print(model_training.initiate_model_training(train_arr, test_arr))
    r2_score_value, best_model_name = model_training.initiate_model_training(train_arr, test_arr)
    print(f"Best Model: {best_model_name}, R2 Score: {r2_score_value}")
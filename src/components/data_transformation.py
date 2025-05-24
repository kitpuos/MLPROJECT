
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig():
    """
    Configuration class for data preprocessing.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Initializes the DataTransformation class and sets up the configuration.
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def data_transformation_object(self):
        logging.info("Entering the data transformation method or component.")

        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            ## Pipelines
            numerical_pipeline = Pipeline(
                steps = [
                    ('Imputer', SimpleImputer(strategy = 'median')),
                    ('Scaler', StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps = [
                    ('Imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('One_Hot_Encoder', OneHotEncoder(sparse_output = False)),
                    ('Scaling', StandardScaler(with_mean = False))
                ]
            )

            logging.info('Numerical columns standard scaling completed.')
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info('Categorical columns encoding completed.')
            logging.info(f"Categorical columns: {categorical_columns}")

            ## Combining the numerical and categorical pipelines

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read the train and test data.')

            logging.info('Obtaining preprocessing object.')
            preprocessing_obj = self.data_transformation_object()

            ## Defining input and target features

            target_column = 'math_score'

            input_features_train_df = train_df.drop(columns = [target_column], axis = 1)
            target_feature_train_df = train_df[target_column]
            
            input_features_test_df = test_df.drop(columns = [target_column], axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object on training and testing dataframes.')

            ## Fitting and transforming data

            input_features_train = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test = preprocessing_obj.fit_transform(input_features_test_df)

            ## Converting into numpy arrays

            train_arr = np.c_[input_features_train, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test, np.array(target_feature_test_df)]

            logging.info('Preprocessing object saved.')

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
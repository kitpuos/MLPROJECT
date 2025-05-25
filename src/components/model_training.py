import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)

@dataclass
class ModelTrainingConfig:
    """
    Configuration class for model training.
    """
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTraining:
    def __init__(self):
        """
        Initializes the ModelTraining class and sets up the configuration.
        """
        self.model_training_config = ModelTrainingConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Entering the model training method or component.")

            logging.info("Splitting the training and testing arrays into features and target variable.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose = False),
                'XGBoost Regressor': XGBRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Extra Trees Regressor': ExtraTreesRegressor()
            }

            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)
            logging.info("Model evaluation completed.")

            ## Get the best model based on R2 score            
            best_model_score = max(sorted(model_report.values()))

            ## Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            ## Threshold for best model score based on R2 score
            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score above threshold.")

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path = self.model_training_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = round(r2_score(y_test, predicted), 4)

            return r2, best_model_name
        
        except Exception as e:
            raise CustomException(e, sys)
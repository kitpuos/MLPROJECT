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

            params = {
                "Linear Regression" : {},
                "KNeighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Decision Tree Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "XGBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 9]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting Regressor": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Extra Trees Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }

            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                               models = models, params = params)
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
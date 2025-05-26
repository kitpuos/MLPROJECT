import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = {}

        for name, model in models.items():
            
            logging.info(f"Evaluating model: {name}")
            
            try:
                param = params.get(name, {})
                if param:
                    logging.info(f"Starting hyperparameter tuning for {name} with params: {param}")
                    gs = GridSearchCV(model, param, cv = 3)
                    gs.fit(X_train, y_train)
                    best_params = gs.best_params_
                    logging.info(f"Best parameters for {name}: {best_params}")
                    model.set_params(**gs.best_params_)
            except Exception as e:
                logging.warning(f"Error in hyperparameter tuning for {name}: {e}")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[name] = round(r2, 3)

        return model_report
        
    except Exception as e:
        raise CustomException(e, sys)
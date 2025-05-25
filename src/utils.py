import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        model_report = {}

        # for i in range(len(list(models))):
        #     model = list(models.values())[i]
        #     model.fit(X_train,y_train)
        #     y_pred = model.predict(X_test)
        #     model_score = r2_score(y_test, y_pred)

        #     model_report[list(models.keys())[i]] = model_score

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[name] = round(r2, 3)

        return model_report


        return model_report
    
    except Exception as e:
        raise CustomException(e, sys)
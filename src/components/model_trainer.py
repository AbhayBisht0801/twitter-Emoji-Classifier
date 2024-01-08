from sklearn.model_selection import train_test_split
import os 
import sys
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_Config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_data,test_data):
        try:
            logging.info('Spliting training and test input data')
        
            X_train=train_data.drop(columns='Label')
            y_train=train_data['Label']
            X_test=test_data.drop(columns='Label')
            y_test=test_data['Label']

            
            models={
                'RandomForest':RandomForestClassifier(),
                'Xgboost':XGBClassifier(),
                'Catboost':CatBoostClassifier(),
                'GaussianNB':GaussianNB(),
                'GradientBoosting':GradientBoostingClassifier(),
                "KNN":KNeighborsClassifier(),
                "mulitNB":MultinomialNB()
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            logging.info('Best found model both training and testing datatset')
            save_object(file_path=self.model_trainer_Config.train_model_file_path,
                        obj=best_model)
            predicted=best_model.predict(X_test)
            accuracy=accuracy_score(y_test,predicted)
            
            return accuracy,best_model
        except Exception as e:
            raise CustomException(e,sys)



import sys
from  dataclasses import dataclass
import pandas as pandas
import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import logging
import os
import nltk
import pandas as pd
from nltk import word_tokenize
nltk.download('punkt')
from langdetect  import detect
from src.utils import clean_data,remove_punctuation,detect_language,remove_tag,save_object
import pickle

tf=TfidfVectorizer(max_features=5000)
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessed.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.tf=None
    
    def fit_transform(self, train_data):
        try:

            train_data['num_chars'] = train_data['TEXT'].apply(len)
            train_data['num_words'] = train_data['TEXT'].apply(lambda x: len(word_tokenize(x)))
            train_data['num_sen'] = train_data['TEXT'].apply(lambda x: len(nltk.sent_tokenize(x)))
            train_data['TEXT'] = train_data['TEXT'].apply(remove_tag)
            train_data['Language'] = train_data['TEXT'].apply(detect_language)
           
            training_data=train_data[train_data['Language']=='en']
            training_data=training_data.reset_index()
            print(training_data.shape)
            training_data['Preprocessed_Data'] = training_data['TEXT'].apply(remove_punctuation)
            training_data['Preprocessed_Data'] = training_data['Preprocessed_Data'].apply(clean_data)
            print(training_data['Label'].isnull().sum())
            self.tf = TfidfVectorizer(max_features=5000)  # Assuming tf is TfidfVectorizer
            vectorized_data = self.tf.fit_transform(training_data['Preprocessed_Data']).toarray()
            print(vectorized_data.shape)
            complete_data = pd.DataFrame(vectorized_data)
            complete_data[['num_words', 'num_chars', 'Label']] = training_data[['num_words', 'num_chars', 'Label']]
        
            complete_data.columns=complete_data.columns.astype(str)
            print(complete_data['Label'].isnull().sum())
            
            file_path = 'artifacts/preprocessed.pkl'

# Open the file in binary write mode ('wb')
            with open(file_path, 'wb') as file:
    # Use the pickle.dump() function to save the data to the file
                pickle.dump(self.tf, file)

            

            # Save the transformer using pickle
            
            return complete_data
        except Exception as e:
            raise CustomException(e, sys)
    def transform(self, test_data):
        try:
            test_data['num_chars'] = test_data['TEXT'].apply(len)
            test_data['num_words'] = test_data['TEXT'].apply(lambda x: len(word_tokenize(x)))
            test_data['num_sen'] = test_data['TEXT'].apply(lambda x: len(nltk.sent_tokenize(x)))
            test_data['TEXT'] = test_data['TEXT'].apply(remove_tag)
            test_data['Language'] = test_data['TEXT'].apply(detect_language)
            testing_data=test_data[test_data['Language']=='en']
            testing_data['Preprocessed_Data'] = testing_data['TEXT'].apply(remove_punctuation)
            testing_data['Preprocessed_Data'] = testing_data['Preprocessed_Data'].apply(clean_data)
            testing_data=testing_data.reset_index()

            

            if self.tf is None:
                raise CustomException("TFIDF Transformer not found. Call fit_transform to fit and save the transformer.", sys)
            vectorized_data = self.tf.transform(testing_data['Preprocessed_Data']).toarray()
            complete_data = pd.DataFrame(vectorized_data)
            complete_data[['num_words', 'num_chars', 'Label']] = testing_data[['num_words', 'num_chars', 'Label']]
            complete_data.columns=complete_data.columns.astype(str)
            print(complete_data['Label'].isnull().sum())
            return complete_data
    
        except Exception as e:
          raise CustomException(e, sys)
    
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('Obtaining the preprocessed object')
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)
            data_transformer=DataTransformation()
            train_preprocess = data_transformer.fit_transform(train)
            test_preprocessed=data_transformer.transform(test)    
            logging.info(f'Preprocessing completed on Data')
           
            return(
                train_preprocess,
                test_preprocessed,
                self.tf
                )
        except Exception as e:
            raise CustomException(e,sys)

        
            



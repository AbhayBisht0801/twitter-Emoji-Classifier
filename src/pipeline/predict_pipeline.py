import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.utils import remove_punctuation,clean_data
from nltk import word_tokenize


df=pd.read_csv('C:\\Users\\bisht\\OneDrive\\Desktop\\END TOend\\artifacts\\emojis.csv')
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessed.pkl'
            model=load_object(file_path=model_path)
            tfid=load_object(file_path=preprocessor_path)
            features['num_chars'] = features['TEXT'].apply(len)
            features['num_words'] = features['TEXT'].apply(lambda x: len(word_tokenize(x)))
            features['Preprocessed']=features['TEXT'].apply(remove_punctuation)
            features['Preprocessed']=features['Preprocessed'].apply(clean_data)
            vectorized_data=tfid.transform(features['Preprocessed']).toarray()
            complete_data = pd.DataFrame(vectorized_data)
            
            complete_data[['num_words', 'num_chars']] = features[['num_words', 'num_chars']]
            complete_data.columns=complete_data.columns.astype(str)
            result=model.predict(complete_data)
        
            emoji=df[df['number']==result[0][0]]['emoticons']
            result= f"{features['TEXT'][0]} {emoji.tolist()[0]}"
            return result
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,text:str):
        self.text=text

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
            "TEXT":[self.text]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            CustomException(e,sys)




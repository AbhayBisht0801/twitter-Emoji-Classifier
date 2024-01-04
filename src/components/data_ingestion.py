import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    emoji_data_path:str=os.path.join('artifacts','emojis.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df=pd.read_csv(r'notebook\\Data\\Train.csv')
            emojis=pd.read_csv(r'notebook\\Data\\Mapping.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path))
            Labels=df['Label'].value_counts()[(df['Label'].value_counts() > 3000) & (df['Label'].value_counts() < 7500)].index.tolist()
            new_df=df[df['Label'].isin(Labels)]
            label_counts =df['Label'].value_counts()
            sorted_labels = label_counts.index.tolist()
            label_mapping = {old_label: new_label-1 for new_label, old_label in enumerate(sorted_labels)}
            new_df['Label']=new_df['Label'].map(label_mapping)
            train_set,test_set=train_test_split(new_df,test_size=0.1,random_state=42)
            emojis=emojis[emojis['number'].isin(Labels)]
            emojis['number']=emojis['number'].map(label_mapping)
            logging.info('Read the dataset as Dataframe')
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train Test Split initiated")
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            emojis.to_csv(self.ingestion_config.emoji_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
            return(self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
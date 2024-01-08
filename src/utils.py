from langdetect import detect
import os
import sys
from src.exception import CustomException
import string
import nltk 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import dill
nltk.download('punkt')
import re
import pickle
from sklearn.metrics import accuracy_score
ps=PorterStemmer()
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def remove_tag(data):
    pattern = re.compile('[\n@#]')
    return pattern.sub('', data)


def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'
pun=string.punctuation+'…'+'・・・'+'•'+"“"+"”"+"…"
pun
def remove_punctuation(text):
    translator = str.maketrans("", "", pun)
    return ' '.join(word.translate(translator) for word in text.split())



def clean_data(text):
  data=text.lower()
  data=word_tokenize(data)
  y=[]
  for i in data:
    if i.isalnum():
      y.append(i)
  data=y[:]
  y.clear()
  for words in text:
    y.append(ps.stem(words))
  return ''.join(y)
def evaluate_model(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)





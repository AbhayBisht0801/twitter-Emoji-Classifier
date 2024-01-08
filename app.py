from flask  import render_template,Flask,request
import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
from sklearn.feature_extraction.text import TfidfVectorizer
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            data=CustomData(
            text=request.form.get('text')
            )
            pred_df=data.get_data_as_dataframe()
            predict_pipeline=PredictPipeline()
          
            result1=predict_pipeline.predict(pred_df)
            
            
            return render_template('home.html',result=result1)
        except Exception as e:
            CustomException(e,sys)
if __name__=="__main__":
    app.run(host="0.0.0.0")
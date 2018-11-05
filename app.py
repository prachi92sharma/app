import json
from flask_cors import CORS
from flask import Flask, request, Response, jsonify
from flask_restful import reqparse, Resource, Api
import os
import json
from sklearn.preprocessing import LabelEncoder
import codecs
from sklearn.externals import joblib
import pandas as pd
import logging
import pyrebase
import numpy as np

app = Flask(__name__)
api = Api(app)
CORS(app)

#Creating an object
logger=logging.getLogger()

#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

#function for machine learning prediction
@app.route('/predict',methods = ['POST'])
def prediction():
    number = LabelEncoder()
    X = request.get_json()
    #print(X)
    #print("JSON INPUT",X,type(X))
    #load the model from disk
    df = pd.read_excel("test.xlsx")
    df['type_0'] = number.fit_transform(df.iloc[:,144].astype('str'))
    #print(df['type_0'])
    df['type_1'] = number.fit_transform(df.iloc[:,145].astype('str'))
    df['plan_0'] = number.fit_transform(df.iloc[:,149].astype('str'))
    #print("!!!!!")
    df['plan_1'] = number.fit_transform(df.iloc[:,150].astype('str'))
    X = df.iloc[0,:]
    loaded_model = joblib.load("model_svm.pkl")
    #print("hii")
    #print("X from df",X)
    X = np.array(X)
    X = np.reshape(X,(1, -1))
    #print(X)
    print(X.shape)
    result = loaded_model.predict(X)
    #print("$%^&*(IO*&^%$%^")
    np_array_to_list = result.tolist()
    #json_file = "file.json" 
    #json.dump(np_array_to_list, codecs.open(json_file, 'w', encoding='utf-8'),separators=(',', ':'), sort_keys=True, indent=4)
    #print("result in json")
    data = {
            'result': np_array_to_list
        }
    res = json.dumps(data)
    resp = Response(res, status=200, mimetype='application/json')
    return resp

if __name__ == '__main__':
    logger.info("Service is up")
    app.run()
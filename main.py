from flask import Flask, render_template
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


##creating a flask app and naming it "app"
app = Flask('app')

acc_ix, hpower_ix, cyl_ix = 2, 4, 0

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]


@app.route('/', methods=['GET'])
def test():
    return 'Pinging Model Application!!'

@app.route('/run', methods=['GET'])
def run():
    print("Hi")
    return render_template("run.html")

@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        print('Hi')
        # form_data = request.form
        # print(form_data)
        vehicle = request.form.to_dict()


        with open('./model_files/model.bin', 'rb') as f_in:
            model = pickle.load(f_in)
            f_in.close()
        with open('./model_files/pipeline.bin', 'rb') as f_in1:
            preproc_pipeline = pickle.load(f_in1)
            f_in1.close()
        predictions = predict_mpg(vehicle, model, preproc_pipeline)

        result = {
            'mpg_prediction': list(predictions)
        }
        #return jsonify(result)
        return render_template('data.html', result = result)
 

# @app.route('/predict', methods=['POST'])
# def predict():
#     vehicle = request.get_json()
#     print(vehicle)
#     with open('./model_files/model.bin', 'rb') as f_in:
#         model = pickle.load(f_in)
#         f_in.close()
#     predictions = predict_mpg(vehicle, model, preproc_pipeline)

#     result = {
#         'mpg_prediction': list(predictions)
#     }
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True, port=9696)
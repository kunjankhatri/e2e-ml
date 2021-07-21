import numpy as np
import pickle
from flask import Flask, render_template, request
from model_files.ml_model import predict_mpg
from sklearn.base import BaseEstimator, TransformerMixin

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
    # return 'Pinging Model Application!!'
    return render_template("run.html")

@app.route('/run', methods=['GET'])
def run():
    print("Hi")
    return render_template("run.html")

@app.route('/predictions', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /predictions is accessed directly. Try going to '/run' to submit data"
    if request.method == 'POST':
        print('Hi')
        # form_data = request.form
        # print(form_data)
        # data to be used to make prediction on
        vehicle = request.form.to_dict()
        
        # load model
        with open('./model_files/model.bin', 'rb') as f_in:
            model = pickle.load(f_in)
            f_in.close()
        # load pipeline
        with open('./model_files/pipeline.bin', 'rb') as f_in1:
            preproc_pipeline = pickle.load(f_in1)
            f_in1.close()
        predictions = predict_mpg(vehicle, model, preproc_pipeline)

        result = {
            'mpg_prediction': round(predictions[0],2)
        }
        #return jsonify(result)
        return render_template('data.html', result = result['mpg_prediction'])

if __name__ == '__main__':
    app.run(debug=True, port=9696)
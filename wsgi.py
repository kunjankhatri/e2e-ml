##importing the app from main file
from main import app
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from main import CustomAttrAdder

acc_ix, hpower_ix, cyl_ix = 2, 4, 0


if __name__ == '__main__': 
    app.run(debug=True)
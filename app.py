from flask import Flask;
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route("/predict/<values>")
def predict(values):
        values = list(map(float,values.split()))
        mlp_from_joblib = joblib.load('MLP.pkl')
        values = np.reshape(values,(1,-1))
        scaler = MinMaxScaler(feature_range=(-1,1))
        ans = mlp_from_joblib.predict(values)
        if ans[0] == 1:
                print(ans[0])
                ret = "rain"
        else:
                print(ans[0])
                ret = "norain"
        return ret
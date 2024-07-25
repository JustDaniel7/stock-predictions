from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm

app = Flask(__name__)

nn_model = tf.keras.models.load_model('stock_price_model.h5')
arima_model = joblib.load('arima_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input']).reshape((1, 50, len(data['input'][0])))

    nn_prediction = nn_model.predict(input_data).tolist()

    arima_prediction = arima_model.forecast(steps=1).tolist()

    return jsonify({'nn_prediction': nn_prediction, 'arima_prediction': arima_prediction})

if __name__ == '__main__':
    app.run(debug=True)

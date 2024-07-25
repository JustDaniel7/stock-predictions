import pandas as pd
import statsmodels.api as sm
import joblib

data = pd.read_csv('preprocessed_data.csv', index_col='Date', parse_dates=True)

model = sm.tsa.ARIMA(data['Close'], order=(5, 1, 0))
model_fit = model.fit()

joblib.dump(model_fit, 'arima_model.pkl')

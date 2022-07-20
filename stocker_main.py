import pandas as pd
import yfinance as yf

# 讀入series
df = yf.Ticker("2330.TW").history(period="max")
price = df.Close
print(price)

from stocker import Stocker

tsmc = Stocker(price)

model, model_data = tsmc.create_prophet_model(days=90)

tsmc.evaluate_prediction()

tsmc.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])

tsmc.predict_future(days=100)


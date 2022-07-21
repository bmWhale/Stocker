import pandas as pd
import yfinance as yf
from stocker import Stocker

# 讀入series
df = yf.Ticker("3034.TW").history(period="max")
price = df.Close

stock = Stocker(price)
model, model_data = stock.create_prophet_model(days=90)
stock.evaluate_prediction()
stock.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
stock.predict_future(days=100)


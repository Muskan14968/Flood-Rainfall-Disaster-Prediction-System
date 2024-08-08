import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt

def daily_runoff_forecast(filename, wtd):
    def import_data():
        raw_data_df = pd.read_excel('sourceCode/data/' + filename + '.xlsx', header=0)
        return raw_data_df

    raw_data_df = import_data()
    raw_data_df['Date'] = pd.to_datetime(raw_data_df['Date'])
    
    data = pd.DataFrame()
    data['Date'] = raw_data_df["Date"]
    data['daily runoff'] = raw_data_df["daily runoff"]
    data = data.set_index(['Date'])

    data.dropna().describe()

    # Resampling
    daily = data.resample('D').sum()

    # Scaling
    values = daily['daily runoff'].values.reshape(-1, 1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scale = daily
    scale["daily runoff"] = scaled

    def making_dataset(i=1):
        df2 = scale.iloc[:, :]
        df2.reset_index(inplace=True)
        df2 = df2.rename(columns={'Date': 'ds', 'daily runoff': 'y'})
        return df2, df2

    df1, df2 = making_dataset(wtd)

    def predicting_data(i=1):
        df2_prophet = Prophet()
        if i == 0:
            df2_forecast = df2_prophet.make_future_dataframe(periods=30*25, freq='D')
            df2_forecast = df2_prophet.predict(df2_forecast)
            df3 = df2_forecast[['ds', 'yhat']]
            df4 = df3.iloc[6940:-20, :]
        else:
            df2_forecast = df2_prophet.make_future_dataframe(periods=30*12, freq='D', include_history=False)
            df2_forecast = df2_prophet.predict(df2_forecast)
            df3 = df2_forecast[['ds', 'yhat']]
            df4 = df3.iloc[:, :]
        return df4, df2_forecast

    df4, df2_forecast = predicting_data(wtd)
    df4.columns = ['Date', 'daily runoff']

    # Plotting the forecast
    df2_prophet.plot(df2_forecast, xlabel='Date', ylabel='daily runoff')
    plt.title('Forecast')
    plt.show()

    return df4

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt

def discharge_forecast(filename, wtd):
    def import_data():
        raw_data_df = pd.read_excel(f'sourceCode/data/{filename}.xlsx', header=0)
        return raw_data_df

    raw_data_df = import_data()
    raw_data_df['Date'] = pd.to_datetime(raw_data_df['Date'])

    for i in range(1, len(raw_data_df.columns)):
        raw_data_df[raw_data_df.columns[i]] = raw_data_df[raw_data_df.columns[i]].fillna(raw_data_df[raw_data_df.columns[i]].mean())

    data = pd.DataFrame()
    data['Date'] = raw_data_df["Date"]
    data['Discharge'] = raw_data_df["Discharge"]
    data = data.set_index(['Date'])

    data.dropna().describe()

    monthly = data.resample('M').sum()
    monthly.plot(style=[':', '--', '-'], title='Monthly Trends')

    weekly = data.resample('W').sum()

    daily = data.resample('D').sum()

    values = daily['Discharge'].values.reshape(-1, 1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scale = daily
    scale["Discharge"] = scaled

    def making_dataset(i=1):
        if i == 0:
            df1 = scale.iloc[6940:, :]
            df2 = scale.iloc[:6940, :]
            df2.reset_index(inplace=True)
            df2 = df2.rename(columns={'Date': 'ds', 'Discharge': 'y'})
            return df1, df2
        else:
            df2 = scale.iloc[:, :]
            df2.reset_index(inplace=True)
            df2 = df2.rename(columns={'Date': 'ds', 'Discharge': 'y'})
            return df2, df2

    df1, df2 = making_dataset(wtd)

    def predicting_data(i=1):
        df2_prophet = Prophet()
        df2_prophet.fit(df2)
        if i == 0:
            df2_forecast = df2_prophet.make_future_dataframe(periods=30 * 25, freq='D')
            df2_forecast = df2_prophet.predict(df2_forecast)
            df3 = df2_forecast[['ds', 'yhat']]
            df4 = df3.iloc[6940:-20, :]
        else:
            df2_forecast = df2_prophet.make_future_dataframe(periods=30 * 12, freq='D', include_history=False)
            df2_forecast = df2_prophet.predict(df2_forecast)
            df3 = df2_forecast[['ds', 'yhat']]
            df4 = df3.iloc[:, :]
        return df4, df2_forecast

    df4, df2_forecast = predicting_data(wtd)
    df4.columns = ['Date', 'Discharge']

    df2_prophet.plot(df2_forecast, xlabel='Date', ylabel='Discharge')
    plt.title('Forecast')
    plt.show()

    return df4

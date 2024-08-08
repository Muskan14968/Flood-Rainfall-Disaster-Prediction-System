import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import prophet as fbprophet  # Correct import name
import matplotlib.pyplot as plt
import joblib

def flood_runoff_forecast(filename, wtd):
    try:
        # Import raw data
        def import_data():
            raw_data_df = pd.read_excel('sourceCode/data/' + filename + '.xlsx', header=0)
            return raw_data_df

        raw_data_df = import_data()
        raw_data_df['Date'] = pd.to_datetime(raw_data_df['Date'])

        for i in range(1, len(raw_data_df.columns)):
            raw_data_df[raw_data_df.columns[i]] = raw_data_df[raw_data_df.columns[i]].fillna(
                raw_data_df[raw_data_df.columns[i]].mean())

        data = pd.DataFrame()
        data['Date'] = raw_data_df["Date"]
        data['flood runoff'] = raw_data_df["flood runoff"]
        data = data.set_index(['Date'])

        # Resampling
        daily = data.resample('D').sum()
        values = daily['flood runoff'].values.reshape(-1, 1)
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        scale = daily
        scale["flood runoff"] = scaled

        # Making dataset for testing or training
        def making_dataset(i=1):
            if i == 0:
                df1 = scale.iloc[6940:, :]
                df2 = scale.iloc[:6940, :]
                df2.reset_index(inplace=True)
                df2 = df2.rename(columns={'Date': 'ds', 'flood runoff': 'y'})
                return df1, df2
            else:
                df2 = scale.iloc[:, :]
                df2.reset_index(inplace=True)
                df2 = df2.rename(columns={'Date': 'ds', 'flood runoff': 'y'})
                return df2, df2

        df1, df2 = making_dataset(wtd)

        # Load or train the model
        path = 'sourceCode/trained/' + filename + '_flood_runoff_prophet'
        df2_prophet = joblib.load(path + '.pkl') if wtd == 0 else fbprophet.Prophet(changepoint_prior_scale=0.05)
        if wtd == 1:
            df2_prophet.fit(df2)
            joblib.dump(df2_prophet, path + '.pkl')

        # Predicting data
        def predicting_data(i=1):
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

        df4, _ = predicting_data(wtd)

        # Inverse scaling
        values = df4['yhat'].values.reshape(-1, 1)
        values = values.astype('float32')
        val = scaler.inverse_transform(values)
        df4['flood runoff'] = val
        df4['flood runoff'] = abs(df4['flood runoff'])
        df4.columns = ['Date', 'flood runoff']

        # Save forecast to CSV
        df4.to_csv('sourceCode/data/forecast/' + filename + '_flood_runoff_forecast.csv', index=False)

        return df4

    except FileNotFoundError:
        print("The file does not exist. Please provide a valid file name.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

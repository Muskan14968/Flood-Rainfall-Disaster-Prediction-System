import discharge_prophet as dp
import flood_runoff_prophet as frp
import daily_runoff_prophet as drp
import weekly_runoff_prophet as wrp
import model
import pandas as pd
import numpy as np
import joblib
import os

def fut_cal(user_date, end_date):
    return 1 if user_date > end_date else 0  # Future Data or Existing Data

def drive(filename, user_date):
    data = pd.read_excel(f'sourceCode/data/{filename}.xlsx')
    user_date = pd.to_datetime(user_date)
    last_date = pd.to_datetime(data['Date'].iloc[-1])
    fut = fut_cal(user_date, last_date)
    check = 0

    if fut == 0:
        for i in range(1, len(data.columns)):
            data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].mean())

        data['Date'] = pd.to_datetime(data['Date'])

        def existing_prediction(i):
            discharge = data.Discharge[i]
            floodrunoff = data['flood runoff'][i]
            dailyrunoff = data['daily runoff'][i]
            weeklyrunoff = data['weekly runoff'][i]
            Flood = data.Flood[i]
            fd = [discharge, floodrunoff, dailyrunoff, weeklyrunoff]

            result, mae = model.flood_classifier(filename, fd)

            discharge = format(round(discharge, 2))
            floodrunoff = format(round(floodrunoff, 2))
            dailyrunoff = format(round(dailyrunoff, 2))
            weeklyrunoff = format(round(weeklyrunoff, 2))
            mae = format(round(mae, 2)) if mae is not None else 'NIL'

            predicted = 'Normal' if result == 0 else 'High'
            actual = 'Normal' if Flood == 0 else 'High'

            print("Features-")
            print(f"Discharge-{discharge} floodrunoff-{floodrunoff} dailyrunoff-{dailyrunoff} weeklyrunoff-{weeklyrunoff}")
            print(f"Actual-{Flood}")
            print(f"Predicted-{result}")
            print(f"Mean-Absolute-Error: {mae}")

            results = {
                "discharge": discharge,
                "floodrunoff": floodrunoff,
                "dailyrunoff": dailyrunoff,
                "weeklyrunoff": weeklyrunoff,
                "meanabsoluteerrorr": mae,
                "predicted": predicted,
                "actualflood": actual
            }
            return results

        for i in range(len(data.Date)):
            if data.Date[i] == user_date:
                results = existing_prediction(i)
                check = 1
                return results

        if check == 0:
            print("Choose a valid date")
    else:
        wtd = 1
        try:
            d1 = dp.discharge_forecast(filename, wtd)
            d2 = frp.flood_runoff_forecast(filename, wtd)
            d3 = drp.daily_runoff_forecast(filename, wtd)
            d4 = wrp.weekly_runoff_forecast(filename, wtd)
        except Exception as e:
            print(f"Error in forecasting: {e}")
            return

        data1 = pd.concat([d1, d2['flood runoff']], axis=1)
        data1 = pd.concat([data1, d3['daily runoff']], axis=1)
        data1 = pd.concat([data1, d4['weekly runoff']], axis=1)

        def future_prediction(i):
            discharge = data1.Discharge[i]
            floodrunoff = data1['flood runoff'][i]
            dailyrunoff = data1['daily runoff'][i]
            weeklyrunoff = data1['weekly runoff'][i]
            fd = [discharge, floodrunoff, dailyrunoff, weeklyrunoff]

            result, mae = model.flood_classifier(filename, fd)

            discharge = format(round(float(discharge), 2))
            floodrunoff = format(round(float(floodrunoff), 2))
            dailyrunoff = format(round(float(dailyrunoff), 2))
            weeklyrunoff = format(round(float(weeklyrunoff), 2))
            mae = format(round(mae, 2)) if mae is not None else 'NIL'

            print("Predicted Features-")
            print(f"Discharge-{discharge} floodrunoff-{floodrunoff} dailyrunoff-{dailyrunoff} weeklyrunoff-{weeklyrunoff}")
            print(f"Predicted-{result}")
            print(f"Mean-Absolute-Error: {mae}")

            predicted = 'Normal' if result == 0 else 'High'

            results = {
                "discharge": discharge,
                "floodrunoff": floodrunoff,
                "dailyrunoff": dailyrunoff,
                "weeklyrunoff": weeklyrunoff,
                "meanabsoluteerrorr": 'NIL',
                "predicted": predicted,
                "actualflood": 'NIL'
            }
            return results

        for i in range(len(data1.Date)):
            if data1.Date[i] == user_date:
                results = future_prediction(i)
                check = 1
                return results

        if check == 0:
            print("Choose a valid future date")

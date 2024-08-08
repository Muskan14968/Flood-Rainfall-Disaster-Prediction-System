import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Flatten
from sklearn.metrics import mean_absolute_error, explained_variance_score

def rainfall(year, region):
    data = pd.read_csv('sourceCode/data/Sub_Division_IMD_2017.csv')
    
    # Separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

    # Fill missing values in numeric columns with the mean
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    # Fill missing values in non-numeric columns with forward fill or another method
    data[non_numeric_cols] = data[non_numeric_cols].fillna(method='ffill')

    data.info()

    def plot_graphs(groundtruth, prediction, title):        
        N = 9
        ind = np.arange(N)  # the x locations for the groups
        width = 0.27       # the width of the bars

        fig = plt.figure(figsize=(18,10))
        fig.suptitle(title, fontsize=12)
        ax = fig.add_subplot(111)
        rects1 = ax.bar(ind, groundtruth, width, color='m')
        rects2 = ax.bar(ind+width, prediction, width, color='c')

        ax.set_ylabel("Amount of rainfall")
        ax.set_xticks(ind+width)
        ax.set_xticklabels(('APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'))
        ax.legend((rects1[0], rects2[0]), ('Ground truth', 'Prediction'))

        for rect in rects1:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                    ha='center', va='bottom')
        for rect in rects2:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                    ha='center', va='bottom')

        plt.savefig('sourceCode/static/img/rainfall.png')

    def data_generation(year, region):
        temp = data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == year]
        data_year = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == region])
        X_year = None
        y_year = None
        for i in range(data_year.shape[1] - 3):
            if X_year is None:
                X_year = data_year[:, i:i + 3]
                y_year = data_year[:, i + 3]
            else:
                X_year = np.concatenate((X_year, data_year[:, i:i + 3]), axis=0)
                y_year = np.concatenate((y_year, data_year[:, i + 3]), axis=0)

        return X_year, y_year

    def data_generation2(region):    
        Kerala = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                  'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['SUBDIVISION'] == region])

        X = None
        y = None
        for i in range(Kerala.shape[1] - 3):
            if X is None:
                X = Kerala[:, i:i + 3]
                y = Kerala[:, i + 3]
            else:
                X = np.concatenate((X, Kerala[:, i:i + 3]), axis=0)
                y = np.concatenate((y, Kerala[:, i + 3]), axis=0)

        return X, y

    def prediction2(year, region):
        # NN model
        inputs = Input(shape=(3, 1))
        x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
        x = Conv1D(128, 2, padding='same', activation='elu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(32, activation='elu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=[inputs], outputs=[x])
        model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])

        X_testing, Y_testing = data_generation(year, region)
        X_train, y_train = data_generation2(region)

        model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.1, shuffle=True)

        y_pred = model.predict(np.expand_dims(X_testing, axis=2))
        mae = mean_absolute_error(Y_testing, y_pred)
        score = explained_variance_score(Y_testing, y_pred)
        
        Y_year_pred = [y_pred[i][0] for i in range(9)]
        y_pred = np.array(Y_year_pred)
        plot_graphs(Y_testing, y_pred, "Year: " + str(year) + '  Region: ' + str(region))
        return mae, score

    mae, score = prediction2(int(year), region)
    mae = format(round(float(mae), 2))
    score = format(round(float(score), 2))
    keras.backend.clear_session()
    return mae, score
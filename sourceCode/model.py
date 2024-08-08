import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib



def flood_classifier(filename, fd, validating=0):
    try:
        data1 = pd.read_excel('sourceCode/data/' + filename + '.xlsx')

        # Preprocess data
        data1['Flood'] = data1['Flood'].apply(lambda x: 1 if x >= 0.1 else 0)
        y = data1['Flood']
        data1.drop(['Flood', 'Date'], axis=1, inplace=True)

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data1 = pd.DataFrame(scaler.fit_transform(data1), columns=data1.columns)

        # Splitting data into training and testing sets
        locate = data1[data1['Year'] == 2015].index.max()
        x_train, y_train = data1.iloc[:locate+1], y.iloc[:locate+1]
        x_test, y_test = data1.iloc[locate+1:], y.iloc[locate+1:]

        # Upsampling the data
        sm = SMOTE(random_state=2)
        x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
        x_train, y_train = shuffle(x_train_res, y_train_res, random_state=0)

        # Load or train the model (Linear Discriminant Analysis)
        path = 'sourceCode/trained/' + filename + '_LDA'
        clf1 = joblib.load(path + '.pkl') if validating else LinearDiscriminantAnalysis()
        if not validating:
            clf1.fit(x_train, y_train)
            joblib.dump(clf1, path + '.pkl')

        # Predictions
        y_predict = clf1.predict(x_test)
        mae = mean_absolute_error(y_test, y_predict)

        return y_predict, mae

    except FileNotFoundError:
        print("The file does not exist. Please provide a valid file name.")
        return None, None
    except Exception as e:
        print("An error occurred:", e)
        return None, None


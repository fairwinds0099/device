import keras

from keras.layers import Sequential
from keras.layers import Dense
from keras.layers import Input

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

if __name__ == '__main__':
    global df
    path = '/Users/apple4u/Desktop/goksel tez/findings_data/2_psy_devcen_email_cen_devavgtime.csv'
    df = pd.read_csv(path)

    df.drop(columns='employee_name', inplace=True)
    df.drop(columns='user_id', inplace=True)
    df.insider_label.fillna(0, inplace=True)
    df.device_avg_duration.fillna(0, inplace=True)

    # print(df.info)
    print(df.head())
    print(df.info)
    X = df.iloc[:, :12].values
    y = df.insider_label.values

    # splitting raw dataset to three: train validation test split and printing sizes
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    # to keep insider/non insider ratio for splitted dataset
    print('=================  data size =========================')
    print('x (raw) size:', len(X), 'y (raw) size:', len(y))
    print('xTrainInitial size:', len(xTrain), 'yTrainInitial size:', len(yTrain))
    xTrain, xVldtn, yTrain, yVldtn = train_test_split(xTrain, yTrain, test_size=0.25, random_state=1, stratify=yTrain)
    print('xTrain size:', len(xTrain), 'yTrainSize:', len(yTrain))
    print('xVal size:', len(xVldtn), 'yValSize:', len(yVldtn))
    print('xTest size:', len(xTest), 'yTest size:', len(yTest))

    reSampler = RandomOverSampler(random_state=0)
    xTrainOverSampled, yTrainOverSampled = reSampler.fit_sample(xTrain, yTrain)
    print('ReSampled xTrain size:', len(xTrainOverSampled), 'Resampled yTrainSize:', len(yTrainOverSampled))

    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    n_bottleneck = n_inputs
    bottleneck = Dense(n_bottleneck)(e)
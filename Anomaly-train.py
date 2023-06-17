from __future__ import print_function
from keras.layers import Convolution1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import CSVLogger
from keras.layers import LSTM
from keras import callbacks
from keras.layers import Convolution1D, Dense, Dropout, MaxPooling1D
from sklearn.preprocessing import Normalizer
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
np.random.seed(1337)  # for reproducibility


# import dataset, first row is header
traindata = pd.read_csv('./UNSW_NB15_training_set.csv', header=0, sep=';', low_memory=False)
testdata = pd.read_csv('./UNSW_NB15_testing_set.csv', header=0, sep=';', low_memory=False)
# traindata.columns = traindata.iloc[0]
# traindata = traindata[1:]
print(traindata.shape)
print(testdata.shape)
print(list(traindata.columns))
for column in traindata.columns:
    try:
        if traindata[column].dtype == type(object):
            le = LabelEncoder()
            traindata[column] = traindata[column].astype(str)
            traindata[column] = traindata[column].str.replace(".", "")
            traindata[column] = le.fit_transform(traindata[column])
    except Exception as e:
        print(e)
        print(column)

for column in testdata.columns:
    try:
        if testdata[column].dtype == type(object):
            le = LabelEncoder()
            testdata[column] = testdata[column].astype(str)
            testdata[column] = testdata[column].str.replace(".", "")
            testdata[column] = le.fit_transform(testdata[column])
    except Exception as e:
        print(e)
        print(column)

X = traindata.iloc[:, 1:44]
Y = traindata.iloc[:, 44]
C = testdata.iloc[:, 44]
T = testdata.iloc[:, 1:44]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
X_test = np.reshape(testT, (testT.shape[0], testT.shape[1], 1))

lstm_output_size = 70

cnn = Sequential()
cnn.add(Convolution1D(64, 3, padding="same",
        activation="relu", input_shape=(43, 1)))
cnn.add(Convolution1D(64, 3, padding="same", activation="relu"))
cnn.add(MaxPooling1D(pool_size=(2)))
cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn.add(MaxPooling1D(pool_size=(2)))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="sigmoid"))


cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(
    filepath="./content/results/cnn3results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='accuracy', mode='max')
csv_logger = CSVLogger(
    './content/results/cnn3results/cnntrainanalysis1.csv', separator=',', append=False)

cnn.fit(X_train, y_train, epochs=43*2, validation_data=(X_test, y_test),
        callbacks=[checkpointer, csv_logger, callbacks.EarlyStopping(monitor='accuracy', patience=3)])
cnn.save("./content/results/cnn3results/cnn_model.hdf5")

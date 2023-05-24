from __future__ import print_function
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score)
from keras.layers import Convolution1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.layers import LSTM
from keras.layers import Convolution1D, Dense, Dropout, MaxPooling1D
from sklearn.preprocessing import Normalizer
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
np.random.seed(1337)  # for reproducibility


testdata = pd.read_csv('./UNSW_NB15_testing_set.csv', header=0, sep=';')
# traindata=pd.read_csv('./UNSW_NB15_training_set.csv', header=0, sep=';')

# for column in traindata.columns:
#     if traindata[column].dtype == type(object):
#         le = LabelEncoder()
#         traindata[column] = traindata[column].astype(str)
#         traindata[column] = traindata[column].str.replace(".", "")
#         traindata[column] = le.fit_transform(traindata[column])

for column in testdata.columns:
    if testdata[column].dtype == type(object):
        le = LabelEncoder()
        testdata[column] = testdata[column].astype(str)
        testdata[column] = testdata[column].str.replace(".", "")
        testdata[column] = le.fit_transform(testdata[column])

# X = traindata.iloc[:,1:44]
# Y = traindata.iloc[:,44]
C = testdata.iloc[:, 44]
T = testdata.iloc[:, 1:44]

# scaler = Normalizer().fit(X)
# trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

# y_train = np.array(Y)
y_test = np.array(C)

# X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
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

# Change this for the path of the weights
cnn.load_weights("results/cnn3results/checkpoint-04.hdf5")


cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

y_pred = cnn.predict_classes(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="binary")
precision = precision_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")
np.savetxt('res/expected3.txt', y_test, fmt='%01d')
np.savetxt('res/predicted3.txt', y_pred, fmt='%01d')

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.6f" % accuracy)
print("racall")
print("%.6f" % recall)
print("precision")
print("%.6f" % precision)
print("f1score")
print("%.6f" % f1)
cm = metrics.confusion_matrix(y_test, y_pred)
print("==============================================")

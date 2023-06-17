import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder

traindata = pd.read_csv('./UNSW_NB15_training_set.csv', header=0, sep=';', low_memory=False)
testdata = pd.read_csv('./UNSW_NB15_testing_set.csv', header=0, sep=';', low_memory=False)

for column in traindata.columns:
    if traindata[column].dtype == type(object):
        le = LabelEncoder()
        traindata[column] = traindata[column].astype(str)
        traindata[column] = traindata[column].str.replace(".", "")
        traindata[column] = le.fit_transform(traindata[column])

for column in testdata.columns:
    if testdata[column].dtype == type(object):
        le = LabelEncoder()
        testdata[column] = testdata[column].astype(str)
        testdata[column] = testdata[column].str.replace(".", "")
        testdata[column] = le.fit_transform(testdata[column])

X1 = traindata.iloc[:, 1:44]
Y1 = traindata.iloc[:, 44]
Y2 = testdata.iloc[:, 44]
X2 = testdata.iloc[:, 1:44]

scaler = Normalizer().fit(X1)
trainX = scaler.transform(X1)

scaler = Normalizer().fit(X2)
testT = scaler.transform(X2)


traindata = np.array(trainX)
trainlabel = np.array(Y1)

testdata = np.array(testT)
testlabel = np.array(Y2)


model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
predictedprob = model.predict_proba(testdata)
np.savetxt('./res/predictedDTprob.txt', predictedprob, fmt='%01d')
np.savetxt('./res/predictedDT.txt', predicted, fmt='%01d')
np.savetxt('./res/expectedDT.txt', expected, fmt='%01d')
# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted, average="binary")
f1 = f1_score(expected, predicted, average="binary")

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)
print("***************************************************************")

'''
@author Shashwat Koranne
'''
# Importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import cluster, svm, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('HRData.csv')

# As we can see, there is non numeric data in the columns, 'sales' and 'salary'.
# We need to convert this text data to numeric values. Following function helps
# us assign numeric keys to the text data.

def convertTextToNumeric(data):
    columns = data.columns.values
    for column in columns:
        convertedValues = {}
        def convertText(text):
            return convertedValues[text]

        if data[column].dtype != np.int64 and data[column].dtype != np.float64:
            columnElements = data[column].values.tolist()
            uniqueElements = set(columnElements)

            num = 0
            for element in uniqueElements:
                if element not in convertedValues:
                    convertedValues[element] = num
                    num = num + 1

            data[column] = list(map(convertText, data[column]))

    return data

# Extracting the X and y columns from the given data.
data = convertTextToNumeric(data)

X = data.drop(['left'], 1)
y = data['left']

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2)

X = np.array(X)
y = np.array(y)

# Let's train different classifiers to apply on our data set.
# Support Vector Machine
# K-Nearest Neighbors
# Logistic Regression

# Training SVM:
svmClassifier = svm.SVC()
svmClassifier.fit(XTrain, yTrain)
svmAccuracy = svmClassifier.score(XTest, yTest) * 100

# Training K-NN
knn = neighbors.KNeighborsClassifier()
knn.fit(XTrain, yTrain)
knnAccuracy = knn.score(XTest, yTest) * 100

# Training Logistic Regression
logReg = LogisticRegression()
logReg.fit(XTrain, yTrain)
logRegAccuracy = logReg.score(XTest, yTest) * 100

Analysis = {}
Analysis['Models'] = ['SVM', 'KNN', 'Log Reg.']
Analysis['Training Error'] = [svmClassifier.score(XTrain, yTrain) * 100, knn.score(XTrain, yTrain) * 100,
                              logReg.score(XTrain, yTrain) * 100]
Analysis['Testing Error'] = [svmAccuracy, knnAccuracy, logRegAccuracy]

Analysis = pd.DataFrame(Analysis)

# We need to predict, which valuable employee will leave next. We can check who
# is most probable to leave by using the trained classifiers.

# Prediction via SVM:
numOfEmployees = 0

for i in range(len(data)):
    if y[i] == 0:
        sample = X[i].reshape(1, -1)
        prediction = svmClassifier.predict(sample)
        if prediction == 1:
            numOfEmployees = numOfEmployees + 1

print "Number of Employees that are likely to leave the job via SVM = ", numOfEmployees

# Prediction via KNN
numOfEmployees = 0

for i in range(len(data)):
    if y[i] == 0:
        sample = X[i].reshape(1, -1)
        prediction = knn.predict(sample)
        if prediction == 1:
            numOfEmployees = numOfEmployees + 1

print "Number of Employees that are likely to leave the job via KNN = ", numOfEmployees

# Prediction via Logistic Regression
numOfEmployees = 0

for i in range(len(data)):
    if y[i] == 0:
        sample = X[i].reshape(1, -1)
        prediction = logReg.predict(sample)
        if prediction == 1:
            numOfEmployees = numOfEmployees + 1

print "Number of Employees that are likely to leave the job via Logistic Regression = ", numOfEmployees

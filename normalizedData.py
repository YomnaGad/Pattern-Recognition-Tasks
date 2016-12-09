import pandas as pd #read files
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

data = pd.read_csv("/home/yomna/patternRecognition/wineShuffle.data", sep=',')
data = data.reindex(np.random.permutation(data.index))
target = data.iloc[:,0]
data= data.iloc[:,1:]

x = int(0.60 * data.shape[0])

#normalization zero mean and standard deviation one
"""standardized_X = preprocessing.scale(data)
X_train = standardized_X[:x]
X_test = standardized_X[x:]"""

#Scaling features in range0,1

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(data)
X_train = X_train_minmax[:x]
X_test = X_train_minmax[x:]

Y_train = target[:x]
Y_test = target[x:]

print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)

gnb = GaussianNB()
Y_pred = gnb.fit(X_train, Y_train).predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)


print("Number of mislabeled points out of a total %d points : %d"
       % (X_test.shape[0],(Y_test != Y_pred).sum()))
print accuracy


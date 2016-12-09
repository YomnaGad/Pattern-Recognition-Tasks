import pandas as pd #read files
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("wineShuffle.data", sep=',')
data = data.reindex(np.random.permutation(data.index))
target = data.iloc[:,0]
data= data.iloc[:,1:]
print(target)
split = 178 * 0.5
X_train = data[:split]
Y_train = target[:split]

X_test = data[split:]
Y_test = target[split:]
print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
gnb = GaussianNB()
Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Number of mislabeled points out of a total %d points : %d"
       % (X_test.shape[0],(Y_test != Y_pred).sum()))
print accuracy

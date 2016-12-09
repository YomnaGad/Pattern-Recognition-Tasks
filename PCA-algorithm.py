import csv
import random
import math
import operator
from sklearn.decomposition import PCA

def doPCA(data):
    pca = PCA(n_components=8)
    pca.fit(data)
    return pca

def loadDataset(filename):
    data = []
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = [[float(y) for y in x] for x in dataset]
        #dataset = map(float, dataset)
        for i in range(len(dataset)):
            data.append(dataset[i][1:])
    return data


data = loadDataset('wineShuffle.data')   
print data
pca = doPCA(data)
print pca.explained_variance_ratio_
transformed_data = pca.transform(data)
print transformed_data
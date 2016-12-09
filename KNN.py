import csv
import random
import math
import operator
from sklearn.metrics import accuracy_score


with open('wineShuffle.data', 'rb') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print ','.join(row)

def loadDataset(filename, split, trainingSet=[], testSet=[], targetTrain=[], targetTest=[]):
    n = split * 178
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            if (x < n):
                trainingSet.append(dataset[x][1:])
                targetTrain.append(float(dataset[x][0]))
            else:
                testSet.append(dataset[x][1:])
                targetTest.append(float(dataset[x][0]))
        print trainingSet

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance) 
 
def getNeighbors(trainingSet, testInstance, k, targetTrain=[]):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist, targetTrain[x]))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append((distances[x][0], distances[x][2]))
    return neighbors 

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(targetTest=[], prediction=[], testSet=[]):
    correct = 0
    for x in range(len(testSet)):
        if(targetTest[x]==prediction[x]):
            correct += 1
    print correct
    print len(testSet)
    size = len(testSet)
    b = float(correct)/float(size) * 100
    return b

def main():
    trainingSet=[]
    targetTest=[]
    targetTrain=[] 
    prediction=[]
    testSet=[]
    split=0.60
    loadDataset('wineShuffle.data', split, trainingSet, testSet, targetTrain, targetTest)
    
    k=5
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k, targetTrain)
        result = float(getResponse(neighbors))
        prediction.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(targetTest[x]))
    accuracy = getAccuracy(targetTest, prediction, testSet)
    print('Accuracy: ' + repr(accuracy) + '%')

main()
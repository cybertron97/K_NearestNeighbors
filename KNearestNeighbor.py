import csv
import math
import random

def readCSV(file):
    data = []
    with open(file, 'r') as fptr:
        handle = csv.reader(fptr)
        for rw in handle:
            if(rw):
                data.append(rw)
    data.pop(0)
    random.shuffle(data)
    return data

def convertStringIntToInt(str):
    return int(str)

def convertStringToInt(dict, str):
    length = len(dict)
    if(not str in dict):
        dict[str] = length
    return dict, length

def compressData(data):
    dicts = []
    for i in range(19):
        dicts.append({})
    for rw in data:
        for i in range(len(rw)):
            if(i == 0 or i == 10 or i == 11 or i == 12 or i == 18):
                rw[i] = convertStringIntToInt(str(rw[i]))
            else:
                dicts[i], length = convertStringToInt(dicts[i], str(rw[i]))
                rw[i] = length
    return data, dicts

def euclideanDistance(takenRows, row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += takenRows[i] * ((row1[i] - row2[i])**2)
    return math.sqrt(distance)

def getKneighbors(trainRows, testRow, takenRows, k):
    dists = []
    kNeighbors = []
    for i in range(len(trainRows)):
        distance = euclideanDistance(takenRows, testRow, trainRows[i])
        dists.append((trainRows[i], distance))
    dists.sort(key=lambda obj: obj[1])
    for i in range(k):
        kNeighbors.append(dists[i])
    return kNeighbors

def getPrediction(kNeighbors):
    count = 0
    for i in range(len(kNeighbors)):
        count += kNeighbors[i][0][len(kNeighbors[i][0])-1]
    if(count > len(kNeighbors)/2):
        return 1
    else:
        return 0
   
def updateResults(expected, actual, currCounts):
    if(expected == actual):
        if(expected == 0):
            currCounts[0][0] += 1
        else:
            currCounts[1][1] += 1
    else:
        if(actual == 0):
            currCounts[0][1] += 1
        else:
            currCounts[1][0] += 1
    return currCounts

def findValue(dict, val):
    if(val in dict):
        return dict[val]
    else:
        return len(dict)/2
   
def transformRow(row, dicts):
    for i in range(len(row)):
        if(i == 0 or i == 10 or i == 11 or i == 12 or i == 18):
            row[i] = convertStringIntToInt(str(row[i]))
        else:
            row[i] = findValue(dicts[i], str(row[i]))
    return row

def removeKFold(data, folds, nfold):
    testData = []
    trainingData = []
    for i in range(len(data)):
        if i >= nfold * len(data)/folds and i < (nfold + 1) * len(data)/folds:
            testData.append(data[i])
        else:
            trainingData.append(data[i])
    return testData, trainingData

csvData = readCSV('fake_job_postings_v4.csv')
compCSVData, dicts = compressData(csvData)

numFolds = 10
for folds in range(numFolds):
    testD, trainingD = removeKFold(compCSVData, 5, folds)
    k = 5
    results = []
    print("Testing With Group ", folds)
    for i in range(len(testD)):
        neighbors = getKneighbors(trainingD, transformRow(testD[i], dicts), [1] * 19, k)
        pred = getPrediction(neighbors)
        results.append(pred)
        #print(i, testD[i][18], pred)

    print("Calculating Results...")
    currCounts = [[0, 0], [0, 0]]
    for i in range(len(results)):
        currCounts = updateResults(testD[i][18], results[i], currCounts)

    precision = 1.0
    if(currCounts[1][1] + currCounts[1][0] > 0):
        precision = currCounts[1][1] * 1.0 / (currCounts[1][1] + currCounts[1][0])

    recall = 1.0
    if(currCounts[1][1] + currCounts[0][1] > 0):
        recall = currCounts[1][1] * 1.0 / (currCounts[1][1] + currCounts[0][1])

    f1Score = 0.0
    if(precision + recall > 0):
        f1Score = (2 * precision * recall) / (precision + recall)

    print("TP: %d" %(currCounts[1][1]))
    print("FP: %d" %(currCounts[1][0]))
    print("TN: %d" %(currCounts[0][0]))
    print("FN: %d" %(currCounts[0][1]))
    print("Precision: %0.4f" %(precision))
    print("Recall   : %0.4f" %(recall))
    print("F1 Score : %0.4f" %(f1Score))
    print()

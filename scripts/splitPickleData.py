import numpy as np
import csv
import cPickle
import gzip
import logging

import sys

#takes our training data and turns it into files suitable for deepnet
rotated = True
rotatedMult = 2 # How many more examples per original example

split = 15 # How many items we split our data into

inputFilename = 'train_inputs.csv'
outputFilename = 'train_outputs.csv'
if rotated :
    inputFilename = 'rotated_' + inputFilename
    outputFilename = 'rotated_' + outputFilename

logging.basicConfig(filename = "logModifyData.log", level = logging.DEBUG, format='%(asctime)s %(message)s')
logging.info("something")

reader = csv.reader(open(inputFilename, 'r'), delimiter=',')
x = list(reader)
arr = np.array(x[1:])[:,1:].astype('float32')
logging.info("arr shape: " + str(np.shape(arr)))

reader = csv.reader(open(outputFilename, 'r'), delimiter=',')
x = list(reader)
labels = np.array(x)[1:,1:].astype('int')
logging.info("labels shape: " + str(np.shape(labels)))

allData = np.concatenate((arr, labels), 1)

print np.shape(allData)

oneTenth = len(allData) / 10

testSet = allData[:oneTenth, :]
validationSet = allData[oneTenth: 3 * oneTenth, :]
trainSet = allData[3 * oneTenth:, :]
#np.random.shuffle(trainSet)
print np.shape(testSet) , np.shape(validationSet), np.shape(trainSet)
if rotated :
    testInputs = testSet[0::rotatedMult, :-1 ]
    testLabels = testSet[0::rotatedMult, -1]
    validationInputs = validationSet[0::rotatedMult, :-1]
    validationLabels = validationSet[0::rotatedMult, -1]
    trainInputs = trainSet[:, :-1]
    trainLabels = trainSet[:, -1]
else :
    testInputs = testSet[:, :-1 ]
    testLabels = testSet[:, -1]
    validationInputs = validationSet[:, :-1]
    validationLabels = validationSet[:, -1]
    trainInputs = trainSet[:, :-1]
    trainLabels = trainSet[:, -1]

print np.shape(testInputs) , np.shape(validationInputs), np.shape(trainInputs)
print np.shape(testLabels) , np.shape(validationLabels), np.shape(trainLabels)
logging.info("dumping")

# Split all arrays into partitions
testInputsSplit = np.array_split(testInputs, split)
testLabelsSplit = np.array_split(testLabels, split)
validationInputsSplit = np.array_split(validationInputs, split)
validationLabelsSplit = np.array_split(validationLabels, split)
trainInputsSplit = np.array_split(trainInputs, split)
trainLabelsSplit = np.array_split(trainLabels, split)

for i in range(split):
    if rotated:
        testInputsFile = gzip.open('digit_test_rotated_split_' + str(i+1) + '_of_' + str(split) + '.pickle.gz', 'wb')
        validationInputsFile = gzip.open('digit_validation_rotated_split_' + str(i+1) + '_of_' + str(split) + '.pickle.gz', 'wb')
        trainInputsFile = open('digit_train_rotated_split_' + str(i+1) + '_of_' + str(split) + '.pickle', 'wb')
    else:
        testInputsFile = gzip.open('digit_test_unrotated_split_' + str(i+1) + '_of_' + str(split) + '.pickle.gz', 'wb')
        validationInputsFile = gzip.open('digit_validation_unrotated_split_' + str(i+1) + '_of_' + str(split) + '.pickle.gz', 'wb')
        trainInputsFile = open('digit_train_unrotated_split_' + str(i+1) + '_of_' + str(split) + '.pickle', 'wb')

    cPickle.dump((testInputsSplit[i].astype('float32'), testLabelsSplit[i]), testInputsFile)
    cPickle.dump((validationInputsSplit[i].astype('float32'), validationLabelsSplit[i]), validationInputsFile)
    cPickle.dump((trainInputsSplit[i].astype('float32'), trainLabelsSplit[i]), trainInputsFile)

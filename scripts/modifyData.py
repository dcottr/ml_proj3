import numpy as np
import csv
import cPickle
import gzip
import logging

#takes our training data and turns it into files suitable for deepnet
numpyForm = False #used for deepnet
pickleForm = True #used for pdnn
rotated = False

inputFilename = 'train_inputs.csv'
outputFilename = 'train_outputs.csv'
if rotated :
    inputFilename = 'rotated_' + inputFilename
    outputFilename = 'rotated_' + outputFilename

logging.basicConfig(filename = "logModifyData.log", level = logging.DEBUG, format='%(asctime)s %(message)s')
logging.info("something")

arr = np.array([np.ones((2305,))])

reader = csv.reader(open(inputFilename, 'r'), delimiter=',')
x = list(reader)
arr = np.array(x)[1:,1:].astype('float32')
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
    testInputs = testSet[0::10, :-1 ]
    testLabels = testSet[0::10, -1]
    validationInputs = validationSet[0::10, :-1]
    validationLabels = validationSet[0::10, -1]
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

if numpyForm :
    fullInputsFile = open('digit_full_inputs.npy', 'w')
    fullLabelsFile = open('digit_full_labels.npy', 'w')
    testInputsFile = open('digit_test_inputs.npy', 'w')
    testLabelsFile = open('digit_test_labels.npy', 'w')
    validationInputsFile = open('digit_validation_inputs.npy', 'w')
    validationLabelsFile = open('digit_validation_labels.npy', 'w')
    trainInputsFile = open('digit_train_inputs.npy', 'w')
    trainLabelsFile = open('digit_train_labels.npy', 'w')
    
    np.save(fullInputsFile, arr)
    np.save(fullLabelsFile, labels)
    np.save(testInputsFile, testInputs)
    np.save(testLabelsFile, testLabels)
    np.save(validationInputsFile, validationInputs)
    np.save(validationLabelsFile, validationLabels)
    np.save(trainInputsFile, trainInputs)
    np.save(trainLabelsFile, trainLabels)

if pickleForm :
#    fullInputsFile = gzip.open('digit_full_rotated.pickle.gz', 'wb')
    testInputsFile = gzip.open('digit_test_rotated.pickle.gz', 'wb')
    validationInputsFile = gzip.open('digit_validation_rotated.pickle.gz', 'wb')
    trainInputsFile = open('digit_train_rotated.pickle', 'wb')

#    cPickle.dump((arr, labels), fullInputsFile)
    cPickle.dump((testInputs.astype('float32'), testLabels), testInputsFile)
    cPickle.dump((validationInputs.astype('float32'), validationLabels), validationInputsFile)
    cPickle.dump((trainInputs.astype('float32'), trainLabels), trainInputsFile)

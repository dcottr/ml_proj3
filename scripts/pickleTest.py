import numpy as np
import csv
import cPickle
import gzip

print "loading test data"
test_X = []
with open('../data/test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for test_input in reader: 
        test_input_no_id = []
        for pixel in test_input[1:]: # Start at index 1 to skip the Id
            test_input_no_id.append(float(pixel))
        test_X.append(test_input_no_id) 

arr = np.array(test_X).astype('float32')

testInputsFile = gzip.open('digit_test.pickle.gz', 'wb')
cPickle.dump((arr.astype('float32'), np.zeros(20000)), testInputsFile)

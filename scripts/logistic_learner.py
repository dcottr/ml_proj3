import csv
import numpy as np

from sklearn import cross_validation, linear_model

print "loading nparrays"
train_X = np.load('../data/train_inputs.npy')
train_Y = np.load('../data/train_outputs.npy')

clf = linear_model.LogisticRegression(verbose=1)
#print cross_validation.cross_val_score(clf, train_X, train_Y, cv=2)
#  [ 0.31289497  0.30614449]

print "training"
clf.fit(train_X, train_Y)

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

print "predicting"
test_Y = clf.predict(test_X)

print "writing precictions"
# Write output
test_output_file = open('../data/test_output_logistic.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',') 
writer.writerow(['Id', 'Prediction']) # write header
for idx, y in enumerate(test_Y):
    row = [idx+1, int(y)]
    writer.writerow(row)
test_output_file.close()

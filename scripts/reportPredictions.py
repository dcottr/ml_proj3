import numpy as np
import sys
import os
import cPickle, gzip
import csv

pred_file = sys.argv[1]

if '.gz' in pred_file:
    pred_mat = cPickle.load(gzip.open(pred_file, 'rb'))
else:
    pred_mat = cPickle.load(open(pred_file, 'rb'))

indices = np.zeros(20000)
for i in range(pred_mat.shape[0]):
    p = pred_mat[i, :]
    p_sorted = (-p).argsort()
    #p_sorted[0] is the index with maximum argument
    indices[i] = p_sorted[0]

indices = np.reshape(indices, (np.shape(indices)[0], 1))
ids = range(np.shape(indices)[0] + 1)[1:]
ids = np.reshape(np.array(ids), (np.shape(ids)[0], 1))

allData = np.concatenate((ids, indices), 1)
allData = allData.astype("int")
labels = np.array([["Id", "Prediction"]])

print np.shape(labels)
print np.shape(allData)

allData = np.concatenate((labels, allData), 0)
allData = allData.astype('string')
print allData[0], allData[1], allData[2]

#np.savetxt("finalOutputs.csv", allData, delimiter=",")

test_output_file = open('finalOutputs.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',') 
writer.writerow(['Id', 'Prediction']) # write header
for idx, y in enumerate(indices):
    row = [idx+1, int(y)]
    writer.writerow(row)
test_output_file.close()

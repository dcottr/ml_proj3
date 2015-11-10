import csv
import numpy as np

from sklearn import cross_validation, svm
from confMatrixDrawer import createAndDrawConfMatrix
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

print "loading nparrays"
X = np.load('../data/train_inputs.npy')
Y = np.load('../data/train_outputs.npy')

#print cross_validation.cross_val_score(clf, train_X, train_Y, cv=2)
#  [ 0.30197584  0.29274342]
# 0.33980 on Kaggle ~ the same as SVM Baseline

numFolds = 4
skf = StratifiedKFold(Y, n_folds = numFolds)

ytruetotal = []
ypredtotal = []

avgTotal = 0
for train_index, test_index in skf:
	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]

	pca = PCA(n_components=1000)
	pca.fit(X_train)
	pcaTrain_X = pca.transform(X_train)
	pcaTest_X = pca.transform(X_test)
	clf = svm.LinearSVC()
	clf.fit_transform(X_train, Y_train)
	Y_pred = clf.predict(X_test)

	ytruetotal.extend(Y_test)
	ypredtotal.extend(Y_pred)
	accuracy =  accuracy_score(Y_test, Y_pred)
	print accuracy
	avgTotal += accuracy

createAndDrawConfMatrix(ytruetotal,  ypredtotal)
print avgTotal/numFolds
# 0.308818796896

'''
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
test_output_file = open('../data/test_output_linearSVM.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',') 
writer.writerow(['Id', 'Prediction']) # write header
for idx, y in enumerate(test_Y):
    row = [idx+1, int(y)]
    writer.writerow(row)
test_output_file.close()
'''
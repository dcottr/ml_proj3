import cPickle, gzip

f1= gzip.open('digit_test.pickle.gz', 'rb')
f1b= gzip.open('digit_test2.pickle.gz', 'wb')
feature, label = cPickle.load(f1)
cPickle.dump((feature, label[:,0]), f1b)
    
f2= gzip.open('digit_full.pickle.gz', 'rb')
f2b= gzip.open('digit_full2.pickle.gz', 'wb')
feature, label = cPickle.load(f2)
cPickle.dump((feature, label[:,0]), f2b)
    
f3= gzip.open('digit_train.pickle.gz', 'rb')
f3b= gzip.open('digit_train2.pickle.gz', 'wb')
feature, label = cPickle.load(f3)
cPickle.dump((feature, label[:,0]), f3b)

f4= gzip.open('digit_validation.pickle.gz', 'rb')
f4b= gzip.open('digit_validation2.pickle.gz', 'wb')
feature, label = cPickle.load(f4)
cPickle.dump((feature, label[:,0]), f4b)
    
import numpy as np
import csv
from matplotlib import pyplot as plt
import random
import scipy.ndimage as ndimage

# Provided code, loads from csv
# Load all training inputs to a python list
train_inputs = []
#with open('../data/train_inputs.csv', 'rb') as csvfile:
with open('../data/rotated_train_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
        for pixel in train_input[1:]: # Start at index 1 to skip the Id
            train_input_no_id.append(float(pixel))
        train_inputs.append(train_input_no_id) 

# Load all training ouputs to a python list
train_outputs = []
#with open('../data/train_outputs.csv', 'rb') as csvfile:
with open('../data/rotated_train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        train_output_no_id = int(train_output[1])
        train_outputs.append(train_output_no_id)

# Faster (much) to load numpy file
#train_inputs = np.load('../data/train_inputs.npy')
#train_outputs = np.load('../data/train_outputs.npy')
c = 0
# Keep displaying random examples until stopped 
while c < 10:
    rand_idx = random.randint(0,len(train_inputs)-1)
    print "Index: %i, Output: %i" % (rand_idx, train_outputs[rand_idx])
    # Convert to numpy array and display as image
    example_input = np.asarray(train_inputs[rand_idx])
    reshaped_input = np.reshape(example_input, (48,48))
    np.save('../paper/image', reshaped_input)
    plt.imshow(reshaped_input, cmap="Greys_r")
    plt.show()
    c += 1

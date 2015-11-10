import numpy as np
import csv
from matplotlib import pyplot as plt
import random
import scipy.ndimage as ndimage
import sklearn

train_inputs = np.load('../data/train_inputs.npy')
train_outputs = np.load('../data/train_outputs.npy')

#transformExamples = 500
spawnRate = 4

new_inputs = open('../data/rotated_train_inputs.csv', "wb")
inputs_writer = csv.writer(new_inputs, delimiter=',')
inputs_writer.writerow(['Id', 'Dim stuff']) # write header

new_outputs = open('../data/rotated_train_outputs.csv', "wb")
outputs_writer = csv.writer(new_outputs, delimiter=',')
outputs_writer.writerow(['Id', 'Prediction']) # write header


index = 1
# Keep displaying random examples until stopped
for idx, train_output in enumerate(train_outputs):
    example_input = np.asarray(train_inputs[idx])
    reshaped_input = np.reshape(example_input, (48,48))
    
    input_row = [index] + example_input.tolist()
    output_row = [index, train_output]
    index += 1
    inputs_writer.writerow(input_row);
    outputs_writer.writerow(output_row);

    usedAngles = set();
    for i in range(spawnRate):
        # Random unique-to-image angle to rotate
        angle = random.randint(1, 359)
        while angle in usedAngles:
            angle = random.randint(1, 359)
        usedAngles.add(angle)
        # Rotate image
        rotated = ndimage.interpolation.rotate(reshaped_input, angle, reshape = False, order = 1, mode = "reflect")
        input_row = [index] + rotated.flatten().tolist()
        output_row = [index, train_output]
        index += 1
        inputs_writer.writerow(input_row);
        outputs_writer.writerow(output_row);


new_inputs.close()
new_outputs.close()

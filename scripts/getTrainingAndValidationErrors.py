# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:46:26 2015

@author: eric
"""

import re
import matplotlib.pyplot as plt

def getTrainingAndValidationErrors(logFileStr):
    logFile = open(logFileStr, "r")
    trainingErrors = []
    validationErrors = []
    for line in logFile :
        if "training error" in line:
            nums = re.findall("\d+\.\d+", line)
            num = nums[1]
            trainingErrors.append(float(num))
        elif "validation error" in line:
            nums = re.findall("\d+\.\d+", line)
            num = nums[2]
            validationErrors.append(float(num))
    
    numCycles = 2 * 15 * 30
    t = range(numCycles)
    print len(t)
    print len(trainingErrors)
    print len(validationErrors)    
    plt.plot(t, trainingErrors[:numCycles], 'bs', t, validationErrors[:numCycles], 'ro')
    plt.show()
    
logFileStr = "/home/eric/School/comp598/assn3/pdnn/rotated 4 times and looped/cnn.training.log"
getTrainingAndValidationErrors(logFileStr)

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 15:33:47 2015

@author: Philippe
"""
import numpy as np
import pandas as pd
import math

class nnLayer:
    neurons = []
    n = 0
    def __init__(self, neurons, n):
        self.n = n
        self.neurons = neurons
        
    def __str__(self):
        return str(self.neurons)
        
    def sigmoid(self, matrix):
        return 1.0/(1 + -np.exp(matrix))
        
    def evaluate(self, w):
        new_neurons = self.sigmoid(-np.dot(self.neurons, w.weights))
        return nnLayer(new_neurons, len(new_neurons))
    
class weights:
    weights = []
    def __init__(self, rows, cols):
        self.weights = np.random.rand(rows, cols)/n
    
if __name__ == "__main__":
    train_X = pd.read_csv('../data/train_inputs.csv')
    train_Y = pd.read_csv('../data/train_outputs.csv')
    train_X = pd.DataFrame.as_matrix(train_X)
    train_Y = pd.DataFrame.as_matrix(train_Y)
    train_X = train_X[:5000,1:]
    n = len(train_X)
    m = len(train_X[0])
    bias = np.ones((n, 1))
    train_X = np.concatenate((train_X, bias), axis=1)
    
    w1 = weights(m + 1, n)
    nnlayer1 = nnLayer(train_X, n)
    nnlayer2 = nnlayer1.evaluate(w1)
    w_end = weights(n, 10)
    nnlayer_end = nnlayer2.evaluate(w_end)
    print nnlayer_end
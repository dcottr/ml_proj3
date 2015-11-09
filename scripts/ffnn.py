# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 15:33:47 2015

@author: Kelley
"""
import numpy as np
import pandas as pd
import math

def sigmoid(matrix, deriv = False):
    if not deriv:
        return 1.0/(1 + np.exp(-matrix))
    else:
        return sigmoid(matrix)*(1 - sigmoid(matrix))

def f_softmax(X, deriv = False):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z
    
class nnLayer:
    def __init__(self, n, m, next_layer_size, is_input=False, is_output=False, activation = sigmoid):
        self.size = n
        self.sigmoid = activation
        self.is_input = is_input
        self.is_output = is_output
        self.weights = None
        self.deltas = None
        self.neurons = None
        self.derivatives = None
        self.tranformed = np.zeros((n, m))
        
        if not is_input:
            self.neurons = np.zeros((n, m))
            self.deltas = np.zeros((n, m))
            
        if not is_output:
            self.weights = np.random.normal(size=(m + 1, next_layer_size), scale=1E-2)
            
        if not is_input and not is_output:
            self.derivatives = np.zeros((m, n))
        
    def __str__(self):
        return str(self.neurons)
        
    def forward(self):
        if self.is_input:
            return self.transformed.dot(self.weights)
            
        self.transformed = self.sigmoid(self.neurons)        
        if self.is_output:
            #self.derivative = self.sigmoid(self.neurons, True).T           
            return self.transformed
        else:
            self.transformed = np.append(self.transformed, np.ones((self.transformed.shape[0], 1)), axis=1)
            self.derivative = self.sigmoid(self.neurons, True).T
            return self.transformed.dot(self.weights)
            
class FFNN:
    def __init__(self, layersizes, n):
        self.layers = []
        self.size = len(layersizes)
        self.layersizes = layersizes
        self.n = n
        
        for i in range(self.size-1):
            if i == 0:
                self.layers.append(nnLayer(n, layersizes[i], layersizes[i + 1], is_input=True))
            else:
                self.layers.append(nnLayer(n, layersizes[i], layersizes[i + 1]))
        
        self.layers.append(nnLayer(n, layersizes[-1], None, is_output = True, activation = f_softmax))
    
    def reset(self):
        for i in range(self.size-1):
            if i == 0:
                self.layers[0] = (nnLayer(self.n, self.layersizes[i], self.layersizes[i + 1], is_input=True))
            else:
                self.layers[i] = (nnLayer(self.n, self.layersizes[i], self.layersizes[i + 1]))
        
        self.layers[self.size - 1] = (nnLayer(self.n, self.layersizes[-1], None, is_output = True, activation = f_softmax))
        
    def forward(self, training):
        self.layers[0].transformed = training
        for i in range(self.size - 1):
            self.layers[i + 1].neurons = self.layers[i].forward()
        return self.layers[-1].forward()
    
    def backward(self, transformed, labels):
        self.layers[-1].deltas = (transformed - labels).T 
        for i in range(self.size - 2, 0, -1):
            nobias = self.layers[i].weights[0:-1, :]
            self.layers[i].deltas = nobias.dot(self.layers[i + 1].deltas) * self.layers[i].derivative
    
    def update_weights(self, eta):
        for i in range(0, self.size-1):
            W_grad = -eta*(self.layers[i+1].deltas.dot(self.layers[i].transformed)).T      
            self.layers[i].weights += W_grad
    
    def evaluate(self, train_data, train_labels, test_data, test_labels, num_epochs=100, eta=0.0001):

        #N_train = len(train_labels)*len(train_labels[0])
        #N_test = len(test_labels)*len(test_labels[0])
        results = []
        print "Training for {0} epochs...".format(num_epochs)
        for t in range(0, num_epochs):
            out_str = "[{0:4d}] ".format(t)

            output = self.forward(train_data)
            self.backward(output, train_labels)
            self.update_weights(eta=eta)

            trainerrs = 0
            output = self.forward(train_data)
            predictions = np.argmax(output, axis=1)
            for i in range(len(train_labels)):
                if train_labels[i][predictions[i]] != 1:
                    trainerrs += 1
            train_percent = float(trainerrs)/len(train_data)
            out_str = "{0} Training error: {1:.5f}".format(out_str, float(trainerrs)/len(train_data))

            testerrs = 0
            output = self.forward(test_data)
            predictions = np.argmax(output, axis=1)
            for i in range(len(test_labels)):
                if test_labels[i][predictions[i]] != 1:
                    testerrs += 1
            #errs += np.sum(1-test_labels[np.arange(len(test_labels)), yhat])
            test_percent = float(testerrs)/len(test_data)
            out_str = "{0} Test error: {1:.5f}".format(out_str, float(testerrs)/len(test_data))
            if(t % 25 == 0):
                results.append((train_percent, test_percent))
            
            print out_str
        return results


if __name__ == "__main__":
    #np.random.seed(1)
    data = pd.read_csv('../data/train_inputs.csv')
    labels = pd.read_csv('../data/train_outputs.csv')
    data = pd.DataFrame.as_matrix(data)
    labels = pd.DataFrame.as_matrix(labels)
    data = data[: , 1:]
    labels = labels[: , 1:]
    '''train_X = data[:5000, 1:]
    train_Y = labels[:5000, 1:]    
    test_X = data[5000:10000, 1:]
    test_Y = labels[5000:10000, 1:]'''#
    
    def randotest(adata,label):
        #adata = np.zeros(np.shape(adata))
        for i in range(len(label)):
            adata[i,10*label[i]:10*label[i]+10]=1
        return adata
    
    def crossvalidate(X, Y, ffnet, alpha):
        total_error = np.zeros(1)
        for i in range(5):
            print "Training on set " + str(i) + " with alpha = " + str(alpha) + " and layersizes = " + str(ffnet.layersizes)
            validationset = X[i * len(X)/5:(i + 1) * len(X)/5]
            trainingset = np.concatenate((X[0 : i * len(X)/5], X[(i + 1) * len(X)/5:]), axis = 0)
            validationlabels = Y[i * len(Y)/5:(i + 1) * len(Y)/5]
            traininglabels = np.concatenate((Y[0 : i * len(Y)/5], Y[(i + 1) * len(Y)/5:]), axis = 0)
            results = ffnet.evaluate(trainingset, traininglabels, validationset, validationlabels, num_epochs = 500, eta=alpha)
            for i in range(len(results)):
                total_error[i] += results[i][1]
            ffnet.reset()
        total_error = total_error/5
        #print str(total_error)
        return total_error
        
    def grid_search(X, Y, num_layers):
        outputs = []
        poss_layersizes = [10, 20, 50, 100, 200, 500, 1000]
        layersizes = [2304]
        for e in range(5):
            alpha = 10**(-(4+e)) 
            if num_layers == 0:
                layersizes.append(10)
                ffnet = FFNN(layersizes, len(X)/5 * 4)
                errors = crossvalidate(X, Y, ffnet, alpha)
                best = 1
                best_idx = 0                
                for i in range(len(errors)):
                    if errors[i] < 1:
                        best = errors[i]
                        best_idx = i
                outputs.append((num_layers, layersizes, alpha, best_idx * 25, best))
                layersizes = [2304]
            elif num_layers == 1:
                for size in poss_layersizes:
                    layersizes.append(20)
                    layersizes.append(10)
                    ffnet = FFNN(layersizes, len(X)/5 * 4)
                    errors = crossvalidate(X, Y, ffnet, alpha)
                    best = 1
                    best_idx = 0                
                    for i in range(len(errors)):
                        if errors[i] < 1:
                            best = errors[i]
                            best_idx = i
                    outputs.append((num_layers, layersizes, alpha, best_idx * 25, best)) 
                    layersizes = [2304]
            elif num_layers == 2:
                for size1 in poss_layersizes:
                    for size2 in poss_layersizes:
                        layersizes.append(size1)
                        layersizes.append(size2)
                        layersizes.append(10)
                        ffnet = FFNN(layersizes, len(X)/5 * 4)
                        errors = crossvalidate(X, Y, ffnet, alpha)
                        best = 1
                        best_idx = 0                
                        for i in range(len(errors)):
                            if errors[i] < 1:
                                best = errors[i]
                                best_idx = i
                        outputs.append((num_layers, layersizes, alpha, best_idx * 25, best))   
                        layersizes = [2304]
        return outputs
              
    ys = np.zeros((50000, 10))
    for i in range(len(labels)):
        ys[i][labels[i][0]] = 1
    bias = np.ones((len(data), 1))
    data = np.concatenate((data, bias), axis=1)
    
    for i in range(3):
        filename = "results" + str(i) + ".txt"
        results = grid_search(X = data, Y = ys, num_layers = 1)
        with open(filename, "w") as f:
            for result in results:
                print str(result)
                f.write(str(result) + "\n")
        
    '''n = len(train_X)
    m = len(train_X[0])
    bias = np.ones((n, 1))
    train_X = np.concatenate((train_X, bias), axis=1)
    test_X = np.concatenate((test_X, bias), axis = 1)
    w1 = weights(m + 1, n)
    nnlayer1 = nnLayer(train_X, n)
    nnlayer2 = nnlayer1.evaluate(w1)
    w_end = weights(n, 10)
    nnlayer_end = nnlayer2.evaluate(w_end)
    mlp = FFNN(layersizes=[2304, 200, 200, 200, 10], n=500)
    mlp.evaluate(train_X, ys, test_X, ys2, eval_train=True)'''
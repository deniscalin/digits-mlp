import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import math
# from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


# READ IN THE DATA
train_data = pd.read_csv('emnist-digits-train.csv')
val_data = pd.read_csv('emnist-digits-test.csv')

# PRINT PD DATA SHAPE
print(f"Train data pd shape: {train_data.shape}, val data pd shape: {val_data.shape}")

# CONVERT DATA TO NUMPY ARRAYS AND SHUFFLE
random.seed(42)
numpy_train_data = np.array(train_data)
numpy_val_data = np.array(val_data)
print("Train data before shuffling: ", numpy_train_data[:, 0])
print("Val data before shuffling: ", numpy_val_data[:, 0])
random.shuffle(numpy_train_data)
random.shuffle(numpy_val_data)
print("Train data after shuffling: ", numpy_train_data[:, 0])
print("Val data after shuffling: ", numpy_val_data[:, 0])

# CREATE X AND Y SPLITS FOR BOTH SETS
print("Creating x and y splits for train and val sets")
x_train = numpy_train_data[:, 1:]
y_train = numpy_train_data[:, :1]

x_val = numpy_val_data[:, 1:]
y_val = numpy_val_data[:, :1]

x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train)

x_val = torch.tensor(x_val, dtype=torch.float)
y_val = torch.tensor(y_val)

print("x_train.shape, y_train.shape, x_val.shape, y_val.shape: ", 
      x_train.shape, y_train.shape, x_val.shape, y_val.shape,)

# TRANSPOSE EACH ROW/IMAGE
print("Transposing each row/image in both train and val sets")
for i in range(x_train.shape[0]):
    x_train[i] = x_train[i].reshape((28, 28)).T.reshape((784))

for i in range(x_val.shape[0]):
    x_val[i] = x_val[i].reshape((28, 28)).T.reshape((784))

# MLP MODEL
# Classes of the model


class Linear:
    
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / (fan_in**0.5)
        self.bias = torch.zeros((fan_out)) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    

class BatchNorm1d:
    
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Trained parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Buffers (trained with momentum update)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self, x):
        # Calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # Update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
    
class Tanh:
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
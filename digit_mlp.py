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
    

# INITIALIZE THE NETWORK

# Removing embeddings
# n_embed = 10 # the dimensionality of the character embedding vectors
n_hidden = 512 # the number of neurons in each layer
vocab_size = 10

g = torch.Generator().manual_seed(42) # Generator for reproducibility

# Removing embeddings
# Setting the input to be 784
# C = torch.randn((vocab_size, n_embed), generator=g)

# 12 Layers
layers = [Linear(784, n_hidden),                  BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden, vocab_size),           BatchNorm1d(vocab_size)
         ]

# Kaiming init
with torch.no_grad():
    #last layer: make less confident
    # layers[-1].weight *= 0.1
    layers[-1].gamma *= 0.1
    
    # for all other layers: multiply by gain for tanh (5/3)
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3
            
parameters = [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True


# TRAINING LOOP

n_steps = 140000
batch_size = 32
lossi = []
ud = []

for i in range(n_steps):
    # Contruct the minibatch
    ix = torch.randint(0, x_train.shape[0], (batch_size,), generator=g)
    Xb, yb = x_train[ix], y_train[ix] # batch X and y
    # Normalize the input
    Xb = (Xb - torch.min(Xb)) / (torch.max(Xb) - torch.min(Xb))
     
    # Forward pass 
    # emb = C[Xb]
    x = Xb.view(ix.shape[0], -1)
    yb = yb.squeeze(1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, yb)
    
    # Backward pass
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # LR SCHEDULE, UPDATING THE PARAMS
    if i < 50000:
        lr = 1e-1
    elif i > 50000 and i < 70000:
        lr = 1e-2
    elif i > 70000 and i < 100000:
        lr = 1e-3
    elif i > 100000:
        lr = 3e-4
    # lr = 1e-1 if i < 40000 else 1e-3
    for p in parameters:
        p.data += -lr * p.grad
        
    # Track stats
    if i % 500 == 0: # print every 10,000 steps
        print(f'lr: {lr:.4f} | {i:7d}/{n_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.item())
    with torch.no_grad():
        ud.append([(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters])

        
# Put all batchnorm layers in inference mode
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        print('Setting the BatchNorm1d layer in inference mode')
        layer.training = False


# VALIDATION LOSS EVAL FUNCTION
@torch.no_grad()
def evaluate():
    out = []
    accs = []
    # batch_size = 32
    runs = 40000
    print(f"Starting the eval with {runs} runs on validation data")

    for i in range(runs):
        # Contruct the minibatch
        ix = torch.randint(0, x_val.shape[0], (batch_size,))
        X_valb, y_valb = x_val[ix], y_val[ix] # batch X and y
        # Normalize the input
        X_valb = (X_valb - torch.min(X_valb)) / (torch.max(X_valb) - torch.min(X_valb))

        # Forward pass 
        # emb = C[Xb]
        x = X_valb.view(ix.shape[0], -1)
        # y_valb = y_valb.squeeze(1)
        for layer in layers:
            x = layer(x)
        # Add some scaling?
        # x = x / x.shape[1]
        probs = F.softmax(x, dim=0)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g)
        out.append(ix)
        # print(out)
        # print(y_valb)
        acc = torch.sum(ix == y_valb) / batch_size
        # print(f"Accuracy: {acc}")
        accs.append(acc)
        
    mean_accuracy = sum(accs) / len(accs)
    print(f"Training run config: {n_steps} steps, {lr} final lr")
    print(f"Mean accuracy over {runs} runs: {mean_accuracy}")


evaluate()


# PLOTTING THE TRAIN LOSS
# plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))


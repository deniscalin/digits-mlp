import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import math
import time
# from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


# CONFIGURE THE DEVICE
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")
device = 'cpu'
print(f"Using device: {device}")

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

x_train = torch.tensor(x_train, dtype=torch.float, device=device)
y_train = torch.tensor(y_train, device=device)

x_val = torch.tensor(x_val, dtype=torch.float, device=device)
y_val = torch.tensor(y_val, device=device)

print("x_train.shape, y_train.shape, x_val.shape, y_val.shape: ", 
      x_train.shape, y_train.shape, x_val.shape, y_val.shape,)

# TRANSPOSE EACH ROW/IMAGE
print("Transposing each row/image in both train and val sets")
for i in range(x_train.shape[0]):
    x_train[i] = x_train[i].reshape((28, 28)).T.reshape((784))

for i in range(x_val.shape[0]):
    x_val[i] = x_val[i].reshape((28, 28)).T.reshape((784))

# print("x_train[0]", x_train[0])
# print("y_train[0]", y_train[0])

# import sys; sys.exit(0)




# MLP MODEL
# Classes of the model


class Linear:
    
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g, device=device) / (fan_in**0.5)
        self.bias = torch.zeros((fan_out), device=device) if bias else None
        
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
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
        # Buffers (trained with momentum update)
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)
        
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
    

class ReLU:

    def __call__(self, x):
        self.out = torch.relu(x)
        return self.out
    
    def parameters(self):
        return []
    



# INITIALIZE THE NETWORK

# Removing embeddings
# n_embed = 10 # the dimensionality of the character embedding vectors
n_hidden = 512 # the number of neurons in each layer
vocab_size = 10

global_seed = 42
print(f"Setting the RNG seed to {global_seed}")
# g = torch.Generator().manual_seed(global_seed) # Generator for reproducibility
# Set global and generation seeds for RNGs
g = torch.manual_seed(global_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(global_seed)
elif torch.backends.mps.is_available():
    g = torch.mps.manual_seed(global_seed)




# Removing embeddings
# Setting the input to be 784
# C = torch.randn((vocab_size, n_embed), generator=g)

# # 8 Layers
# layers = [Linear(784, n_hidden),                  BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, n_hidden),             BatchNorm1d(n_hidden), Tanh(),
#           Linear(n_hidden, vocab_size),           BatchNorm1d(vocab_size)
#          ]

layers = [
            Linear(784, 512),       BatchNorm1d(512), Tanh(),
            Linear(512, 256),       BatchNorm1d(256), Tanh(),
            Linear(256, 128),       BatchNorm1d(128), Tanh(),
            Linear(128, 64),        BatchNorm1d(64),  Tanh(),
            Linear(64, vocab_size), BatchNorm1d(vocab_size)
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

n_steps = 300000
batch_size = 128
lossi = []
ud = []


# VALIDATION LOSS EVAL FUNCTION
@torch.no_grad()
def evaluate(runs):
    accs = []
    losses = []
    print(f"Starting the eval with {runs} runs on validation data")

    for i in range(runs):
        # Contruct the minibatch
        idx = torch.randint(0, x_val.shape[0], (batch_size,))
        X_valb, y_valb = x_val[idx], y_val[idx] # batch X and y
        # Normalize the input
        X_valb = (X_valb - torch.min(X_valb)) / (torch.max(X_valb) - torch.min(X_valb))

        # Forward pass 
        # emb = C[Xb]
        x_eval = X_valb.view(idx.shape[0], -1)
        # y_valb = y_valb.squeeze(1)
        for layer in layers:
            x_eval = layer(x_eval)
        # Add some scaling?
        # x = x / x.shape[1]
        probs_eval = F.softmax(x_eval, dim=0)
        idx = torch.multinomial(probs_eval, num_samples=1, replacement=True, generator=g)
        acc = torch.sum(idx == y_valb) / batch_size
        y_valb = y_valb.squeeze(1)
        loss = F.cross_entropy(x_eval, y_valb)
        # print(f"Accuracy: {acc}")
        accs.append(acc)
        losses.append(loss)
        
    mean_accuracy = sum(accs) / len(accs)
    mean_loss = sum(losses) / len(losses)
    # print(f"Training run config: {n_steps} steps, {lr} final lr")
    print(f"Mean accuracy over {runs} runs on val data: {mean_accuracy}")
    print(f"Mean loss over {runs} runs on val data: {mean_loss}")
    return mean_accuracy, mean_loss


for i in range(n_steps):
    t0 = time.time()
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
    if i < 100000:
        lr = 5e-3
    elif i > 100000 and i < 120000:
        lr = 4e-3
    elif i > 120000 and i < 140000:
        lr = 3e-3
    elif i > 140000 and i < 160000:
        lr = 2e-3
    elif i > 160000 and i < 180000:
        lr = 1e-3
    elif i > 180000 and i < 200000:
        lr = 9e-4
    elif i > 200000 and i < 220000:
        lr = 8e-4
    elif i > 220000 and i < 240000:
        lr = 7e-4
    elif i > 240000 and i < 260000:
        lr = 6e-4
    elif i > 260000 and i < 280000:
        lr = 5e-4
    elif i > 280000 and i < 300000:
        lr = 4e-4
    # elif i > 90000 and i < 160000:
    #     lr = 5e-3
    # elif i > 130000:
    #     lr = 4e-4
    # lr = 1e-1 if i < 40000 else 1e-3
    for p in parameters:
        p.data += -lr * p.grad

    # Track stats
    t1 = time.time()
    dt = (t1 - t0) # Time delta in seconds
    if i % 500 == 0: # print every 10,000 steps
        print(f'For step: {i:7d}/{n_steps:7d} | loss: {loss.item():.4f} | lr: {lr:.4f} | dt: {dt*1000:.4f} s')
    lossi.append(loss.item())
    with torch.no_grad():
        ud.append([(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters])

    # Run evals
    # Evaluate on val data every 10000 steps
    if i != 0 and i % 10000 == 0:
        evaluate(runs=5000)
        

# # Put all batchnorm layers in inference mode
# for layer in layers:
#     if isinstance(layer, BatchNorm1d):
#         print('Setting the BatchNorm1d layer in inference mode')
#         layer.training = False


evaluate(runs=40000)


# PLOTTING THE TRAIN LOSS
# plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))

# SAVING THE MODEL WEIGHTS
checkpoint_path = 'model_' + str(n_steps) + '.pt'

checkpoint = {
    "layers": layers,
    "parameters": parameters,
    "n_steps": n_steps
}

torch.save(checkpoint, checkpoint_path)
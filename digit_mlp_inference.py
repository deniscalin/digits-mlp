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
from utils import display_images, prepare_image


train = True

# CONFIGURE THE DEVICE
if train:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    device = 'cpu'
    print(f"Using device: {device}")
device = 'cpu'

# SETTING GLOBAL RNG SEED
global_seed = 42
print(f"Setting the RNG seed to {global_seed}")
# g = torch.Generator().manual_seed(global_seed) # Generator for reproducibility
# Set global and generation seeds for RNGs
g = torch.manual_seed(global_seed)

# READ IN THE DATA
if train:
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


    # np.save('x_train_0.npy', x_train[0])
    # torch.save(x_train[0], 'test_images/x_train_0.pt')


# MLP
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
        self.training = False
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
    

loaded_model = torch.load('model_300000.pt')
layers = loaded_model['layers']
parameters = loaded_model['parameters']

batch_size = 128


# Set BatchNorm layers to inference mode
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        print(layer)
        layer.training = False
        print("Set BatchNorm1D training to false")
    print(layer)

# img_tensor = torch.load('test_images/img.pt')
# print("Shape: ", img_tensor.shape)
# print("Loaded tensor: ", img_tensor)
# print("Reshaped tensor: ", img_tensor.reshape((28, 28)))

# COMMENTING OUT FOR TESTING
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


@torch.no_grad()
def inference_function(x_batch):
    """The inference function, which takes the input tensor of 1 image, 
    and returns the label prediction and confidence score for this prediction.
    
    Args:
        x_batch | input torch tensor vector of shape [784]"""
    # Normalize the input
    x_batch = (x_batch - torch.min(x_batch)) / (torch.max(x_batch) - torch.min(x_batch))
    # print("x_batch shape: ", x_batch.shape)
    # x_batch = x_batch.view(idx.shape[0], -1)
    # print("x_batch shape after reshape: ", x_batch.shape)
    for layer in layers:
        x_batch = layer(x_batch)
    # print('Logits: ', x_batch)
    # print('Logits shape: ', x_batch.shape)
    probs = F.softmax(x_batch, dim=-1)
    # print('Probabilities: ', probs)
    # ix = probs.argmax()
    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g)
    # print("probs shape: ", probs.shape)
    # print("ix shape: ", ix.shape)
    # print("ix: ", ix)
    conf_score = probs[0][ix]
    # print("conf_score shape: ", conf_score.shape)
    ix = ix[0][0].item()
    conf_score = conf_score[0][0].item() * 100 
    # print("Prediction: ", ix)
    return ix, conf_score


img_tensor = prepare_image('test_images/IMG_4940.HEIC')
# Load an image
# img_tensor = torch.load('test_images/img.pt')
# img_tensor = torch.load('test_images/x_train_0.pt')
# print("Loaded img_tensor: ", img_tensor)
# idx = torch.tensor([1])

for i in range(50):
    ix, conf_score  = inference_function(x_val[i])
    print("Prediction: ", ix)
    print(f"Confidence score: {conf_score:.2f}%")
    if ix == y_val[i].item():
        print("Pass")
    else:
        print("Fail")

# display_images(x_val[7])



# Comenting out for testing
# idx = torch.arange(0, 32)
# idx = 1
# inference_function(x_val[idx], idx)
# print("Label: ", y_val[idx])

# evaluate(runs=32)

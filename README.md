## This is a Multilayer Perceptron neural network implemented in 4 basic classes (Linear, BatchNorm1D, Tanh, ReLU) on top of low-level PyTorch API (torch tensors).

### Training data
This model is trained on the expanded EMNIST digits dataset (240K training samples, 40K validation samples) of handwritten digits.

### Results
Currently, this model achieves a validation data accuracy score of 0.975 (mean accuracy over 40000 random batches of size 128 drawn from the val dataset).
## Handwritten Digits Recognition MLP
This is a Multilayer Perceptron neural network implemented in 4 basic classes (Linear, BatchNorm1D, Tanh, ReLU) on top of low-level PyTorch API (torch tensors, with the exception of using torch.tanh() and torch.relu()). This low-level implementation in pure torch.tensors has been inspired by [Andrej Karpathy's amazing series of videos](https://www.youtube.com/watch?v=P6sfmUTpUmc), which I re-implemented and adapted for sequence classification and digit recognition. 

### Training data
this model has been trained on the custom expanded MNIST Digits [train dataset 240k rows by 785 columns](https://www.kaggle.com/datasets/deniscalin/emnist-digits?select=emnist-digits-train.csv) and evaluated on the [eval dataset 40k rows by 785 columns](https://www.kaggle.com/datasets/deniscalin/emnist-digits?select=emnist-digits-test.csv).

### Results
Currently, this model achieves a validation data accuracy score on unseen data of `97.5%` (mean accuracy over 40000 random batches of size 128 drawn from the val dataset).
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# @dataclass
# class MLPConfig:
#     size: list = field(default_factory=list)
#     epochs: int = 10
#     lr: float = 1e-3


class MLP:
    
    def __init__(self, sizes, epochs, lr):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr
        
        input_layer = sizes[0]
        hidden_layer_1 = sizes[1]
        hidden_layer_2 = sizes[2]
        output_layer = sizes[3]
        
        self.params = {
            "W1": np.random.randn(input_layer, hidden_layer_1),      # 768, 128
            "W2": np.random.randn(hidden_layer_1, hidden_layer_2),   # 128, 64
            "W3": np.random.randn(hidden_layer_2, output_layer)      # 64, 10
        }
    
    def forward(self, x):
        
        w1_out = np.dot()
        
    
    def backward(self):
        pass
    
    def evaluate(self):
        pass


# Get the data
data = pd.read_csv('train.csv')

# Convert to a numpy array
numpy_data = np.array(data)

# Shuffle in place
np.random.shuffle(numpy_data)

# Separate into train and val sets
n = int(len(numpy_data) * 0.8)
train_data = numpy_data[:n]
val_data = numpy_data[n:]

# Create x and y splits for each set
x_train = train_data[:, 1:]
y_train = train_data[:, :1]

x_val = val_data[:, 1:]
y_val = val_data[:, :1]

#
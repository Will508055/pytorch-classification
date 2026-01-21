import torch.nn as nn


class wide_nn(nn.Module):                     # 21,166 parameters
    def __init__(self):
        super().__init__()                    # Inherit the properties of parent class nn.Module
        self.hidden = nn.Linear(83, 249)      # First layer has 83 inputs (number of input features) and 249 outputs (x3)
        self.relu = nn.ReLU()                 # ReLU activation function
        self.output = nn.Linear(249, 1)       # Output layer has 249 inputs and 1 output (for binary classification)
        self.sigmoid = nn.Sigmoid()           # Sigmoid activation for continuous output between 0 and 1
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
    

class deep_nn(nn.Module):                     # 21,000 parameters
    def __init__(self):
        super().__init__()                    # Inherit the properties of parent class nn.Module
        self.layer1 = nn.Linear(83, 83)       # First layer has 83 inputs (number of input features) and 83 outputs
        self.act1 = nn.ReLU()                 # ReLU activation function
        self.layer2 = nn.Linear(83, 83)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(83, 83)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(83, 1)        # Output layer has 83 inputs and 1 output (for binary classification)
        self.sigmoid = nn.Sigmoid()           # Sigmoid activation for continuous output between 0 and 1
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
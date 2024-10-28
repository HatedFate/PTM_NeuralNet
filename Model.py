import torch
import torch.nn as nn

# Defining Neural Network
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.hidden1 = nn.Linear(in_features=3 * 16, out_features=64, dtype=torch.float32)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.hidden2 = nn.Linear(in_features=64, out_features=64, dtype=torch.float32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.hidden3 = nn.Linear(in_features=64, out_features=128, dtype=torch.float32)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.output = nn.Linear(in_features=128, out_features=1, dtype=torch.float32)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.act1(self.batch_norm1(x))
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = self.act2(self.batch_norm2(x))
        x = self.dropout2(x)
        x = self.hidden3(x)
        x = self.act3(self.batch_norm3(x))
        x = self.dropout3(x)
        x = self.act_output(self.output(x))
        return x                                                                                                                                                                              ~                                         

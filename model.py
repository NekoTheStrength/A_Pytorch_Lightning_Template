import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, bias=True):
        super(Network, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.bias = bias

        self.d1 = nn.Linear(self.in_channels, out_channels, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.d2 = nn.Linear(self.out_channels, out_channels, bias=self.bias)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.d3 = nn.Linear(self.out_channels, num_classes, bias=self.bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.in_channels)
        x = self.d1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.d2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.d3(x)
        return x

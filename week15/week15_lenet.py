import torch
import torch.nn as nn
from openpyxl.chart.legend import LegendEntry

from week12 import activation_function


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2),
            nn.Sigmoid()
        )
        self.pooling_1 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Sigmoid()
        )
        self.pooling_2 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Linear(in_features=84, out_features=num_classes)


    def forward (self,x):

        x = self.conv_1(x)
        x = self.pooling_1(x)
        x = self.conv_2(x)
        x = self.pooling_2(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        return x

fake_data = torch.rand(1, 1, 28, 28)
model = NeuralNetwork(num_classes=10)
pre = model(fake_data)
print(pre)


import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU()
        )
        self.pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(96,256, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=256*5*5, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
    def forward(self,x):
        x = self.conv_1(x)
        x = self.pooling_1(x)
        x = self.conv_2(x)
        x = self.pooling_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.pooling_3(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        return x

fake_data = torch.rand(16, 3, 224, 224)
model = NeuralNetwork(num_classes=1000)
pre = model(fake_data)
print(pre)
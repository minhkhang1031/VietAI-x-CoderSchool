import torch
import torch.nn as nn
import torchvision.models as models

class Multilabelresnet(nn.Module):
    def __init__(self):
        super(Multilabelresnet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        number_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc = nn.Linear(in_features=number_feature, out_features=128)
        self.drop_out = nn.Dropout(p=0.3)
        self.number_head = nn.Linear(128,12)
        self.color_head = nn.Linear(128,1)


    def forward(self,x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc(x))
        x = self.drop_out(x)
        jersey_number = self.number_head(x)
        jersey_color = self.color_head(x)

        return jersey_number, jersey_color

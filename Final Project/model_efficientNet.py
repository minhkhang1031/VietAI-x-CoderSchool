import torch
import torch.nn as nn
import torchvision.models as models

class Multilabelefficient(nn.Module):
    def __init__(self):
        super(Multilabelefficient, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        number_feature = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

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

# import torch
# import timm
# import torch.nn as nn
#
# class ViTMultilabel(nn.Module):
#     def __init__(self):
#         super(ViTMultilabel, self).__init__()
#         self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True)
#
#         num_features = self.backbone.head.in_features
#         self.backbone.head = nn.Identity()  # Bỏ lớp classification mặc định
#
#         self.fc = nn.Linear(num_features, 128)
#         self.number_head = nn.Linear(128, 12)  # 12 classes (số áo)
#         self.color_head = nn.Linear(128, 1)  # 1 class (màu áo, BCE loss)
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = torch.relu(self.fc(x))
#         jersey_number = self.number_head(x)
#         jersey_color = self.color_head(x)
#         return jersey_number, jersey_color

from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
import os

class FootballDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_name = self.labels.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        jersey_number = self.labels.iloc[index, 1]
        team_color = self.labels.iloc[index, 2]

        return image, torch.tensor(jersey_number), torch.tensor(team_color)

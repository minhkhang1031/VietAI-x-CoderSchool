import glob
import os
import numpy as np
import torch
import cv2
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

"""
    Dataset các bạn download tại:
    https://drive.google.com/drive/folders/15wG2QgWU8dKs-NeoLI48TxzFWdydg7Jj?usp=drive_link
    Các bạn hãy tạo class Dataset cho bộ data này mà không dùng ImageFolder nhé
"""

class AnimalDataset(Dataset):
    def __init__(self, animals):
        labels = os.listdir(animals)
        self.imgs = []
        self.labels = []

        for label in labels:
            sub_path = os.path.join(animals, label)
            images_files = glob.glob(sub_path + "/*.j*")

            self.imgs.extend(images_files)
            temp =  [label] * len(images_files)
            self.labels.extend(temp)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512,512))
        image = image/image.mean()
        return image, label

if __name__ == '__main__':
    #dataset = CIFARDataset(root="../data")
    animal_path = "./animals/train"
    dataset = AnimalDataset(animals = animal_path)
    index = 200
    image, label = dataset.__getitem__(index)
    print(image.shape)
    print(label)
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
    plt.show()
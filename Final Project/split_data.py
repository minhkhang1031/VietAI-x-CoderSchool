import os
import random
import shutil
import glob
import re

img_dir = "./football_data/images"
label_dir = "./football_data/labels"

#create output
output_dir = "dataset_football"
os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

#get list image
img_list = [img for img in glob.glob(img_dir + "/*")]
random.shuffle(img_list)

#split train/valid
train_rate = int(len(img_list) * 0.8)
train_set = img_list[:train_rate]
valid_set = img_list[train_rate:]

def copy_file(img_dir, type):
    for img in img_dir:
        filename = re.split(r'[\\/]',img)[-1]
        label = filename.replace(".png", ".txt")

        shutil.copy(img, f"{output_dir}/images/{type}/{img}")
        shutil.copy(os.path.join(label_dir,label), f"{output_dir}/labels/{type}/{label}")

copy_file(train_set, "train")
copy_file(train_set, "val")

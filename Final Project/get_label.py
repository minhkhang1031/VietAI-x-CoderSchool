import json, cv2
import os.path
import torch
import glob
from tqdm.auto import tqdm
import pandas as pd
import re


def read_file_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def cal_bbox(bbox, img_w, img_h):
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h


def construct_hash(data):
    id_to_ann = dict()
    for img in tqdm(data["images"]):
        img_id = img["id"]
        anns = []
        for ann in data["annotations"]:
            if ann["image_id"] == img_id and ann["category_id"] in [3, 4]:
                anns.append(ann)
        id_to_ann[img_id] = anns
    return id_to_ann


def create_ann(data, id_to_ann, img, filename):
    img_id = img["id"]
    anns = id_to_ann[img_id]
    ann_normalize = []
    for ann in anns:
        category_id = ann["category_id"]
        if category_id == 4:
            label = 0
        elif category_id == 3:
            label = 1
        else:
            continue
        nor_box = cal_bbox(ann["bbox"], img["width"], img["height"])
        nor_box = [label] + list(nor_box)
        ann_normalize.append(nor_box)

    if ann_normalize:
        df = pd.DataFrame(ann_normalize)
        df.to_csv(filename, header=None, index=None, sep=" ")


def convert_to_yolo(folder_json, output_dir):
    json_path = glob.glob(folder_json + "/*.json")[0]
    data = read_file_json(json_path)
    os.makedirs(output_dir, exist_ok=True)
    hash_map = construct_hash(data)

    for img in tqdm(data["images"]):
        filename = re.split(r'[\\/]', folder_json)[-1] + "_" + img["file_name"]
        filename = os.path.join(output_dir, filename.replace(".PNG", ".txt"))
        create_ann(data, hash_map, img, filename)


json_path = "./Datasets/football_train"
output_dir = "./football_data/labels"
img_dir = "./football_data/images"

list_subfolder = glob.glob(json_path + "/*")

for i in list_subfolder:
    convert_to_yolo(i, output_dir)

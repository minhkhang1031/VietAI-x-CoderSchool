import json
import os
import cv2
import pandas as pd
import glob
import re
import uuid
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# dùng bbox trong file json => get x, y, w, h từ "annotations" (category_id = 4) => get jersey_number and team_jersey_color for label
# cắt hình ảnh theo filename từ "images" (filename = subfolder + filename)

def read_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data

def split_class_jersey_number(jersey_number):
    jersey_number = int(jersey_number)
    if jersey_number >= 11:
        return 11
    elif 1 <= jersey_number <= 10:
        return jersey_number


def normalize_team_jersey_color(team_jersey_color):
    if team_jersey_color == "white":
        return 1
    elif team_jersey_color == "black":
        return 0
    else:
        return -1

def crop_image(folder_path, img_path, outputdir):
    json_path = glob.glob(folder_path + "/*.json")[0]
    data = read_json(json_path)
    subfolder = re.split(r'[\\/]', folder_path)[-1]

    list_player = []
    crop_img = []

    for img_info in tqdm(data["images"]):
        img_id = img_info["id"]
        img_name = subfolder + "_" + img_info["file_name"]
        img = cv2.imread(filename=os.path.join(img_path, img_name))
        if img is None:
            continue

        annot_list = [ann for ann in data["annotations"] if ann["image_id"] == img_id and ann["category_id"] == 4]

        number_0 = ['invisible', 'partially_visible']

        for ann in annot_list:
            if ann['attributes']['occluded'] == "no_occluded":

                x, y, w, h = map(int, ann["bbox"])
                crop_image = img[y:y + h, x:x + w]

                if ann['attributes']['number_visible'] in number_0:
                    player_number = 0
                else:
                    player_number = ann['attributes']['jersey_number']
                    player_number = split_class_jersey_number(player_number)

                player_color = ann['attributes']['team_jersey_color']
                player_color = normalize_team_jersey_color(player_color)

                if player_color == -1:
                    continue

                img_name = f"player_{uuid.uuid4().hex}.png"
                crop_img.append(crop_image)
                list_player.append((img_name, player_number, player_color))

                crop_path = os.path.join(outputdir, img_name)
                cv2.imwrite(crop_path, crop_image)

    return list_player

path = "Datasets/football_train/Match_1864_1_0_subclip/Match_1864_1_0_subclip.json"
img_path = "dataset_football/images/train"
json_path = "Datasets/football_test/Match_1824_1_0_subclip_3"
output_dir = "player_new"
player = crop_image(json_path, img_path, output_dir)

# img_path = "dataset_football/images/train"
# output_dir = "player_new"
# os.makedirs(output_dir, exist_ok=True)
# list_player = []
# list_json = glob.glob("Datasets/football_train" + "/*")
# for json_path in list_json:
#     players = crop_image(json_path, img_path, output_dir)
#     list_player.extend(players)

df = pd.DataFrame(player, columns=["filename", "jersey_number", "team_jersey_color"])
df.to_csv("player_val.csv", index=None, sep=",")
print("Ghi thanh cong!")

# df_split = pd.read_csv("player_visible.csv")
# train_df, val_df = train_test_split(df_split, train_size=0.8, random_state=12, stratify=df_split['jersey_number'])
#
# train_df.to_csv("train_visible.csv", index=None)
# val_df.to_csv("valid_visible.csv", index=None)
# print("Ghi thanh cong!")

"""
📊 Số lượng mẫu theo số áo:
  - Số áo 0: 32874 mẫu (64.02%)
  - Số áo 1: 2085 mẫu (4.06%)
  - Số áo 2: 1839 mẫu (3.58%)
  - Số áo 3: 1761 mẫu (3.43%)
  - Số áo 4: 973 mẫu (1.89%)
  - Số áo 5: 2444 mẫu (4.76%)
  - Số áo 6: 1140 mẫu (2.22%)
  - Số áo 7: 1485 mẫu (2.89%)
  - Số áo 8: 1558 mẫu (3.03%)
  - Số áo 9: 1828 mẫu (3.56%)
  - Số áo 10: 794 mẫu (1.55%)
  - Số áo 11: 2571 mẫu (5.01%)

🎨 Số lượng mẫu theo màu áo:
  - Màu Trắng: 25832 mẫu (50.30%)
  - Màu Đen: 25520 mẫu (49.70%)

🧐 Giá trị duy nhất trong cột màu áo: [1 0]
"""
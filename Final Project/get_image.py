import glob
import json
import os
import cv2
from tqdm import tqdm
import re

# get video => create folder contain farmes => read video => create file name => write file
def create_frame(path_dir, output_dir):
    video = glob.glob(path_dir + "/*.mp4")[0]
    video_folder = re.split(r'[\\/]', path_dir)[-1]
    frame_count = 0
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Extracting Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            file_name = os.path.join(output_dir, f"{video_folder}_frame_{frame_count:06d}.png")
            cv2.imwrite(file_name, frame)
            frame_count += 1
            pbar.update(1)

        cap.release()


# x = "./Datasets/football_train/Match_1951_1_0_subclip"
# create_frame(x)
output_dir = "./football_data/images"
list_folder = glob.glob("./Datasets/football_train" + "/*")
for i in list_folder:
    create_frame(i, output_dir)

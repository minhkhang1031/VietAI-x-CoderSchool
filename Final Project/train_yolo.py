from ultralytics import YOLO
import ultralytics.data.build as build
from dataset_yolo import YOLOWeightedDataset

if __name__ == '__main__':

    build.YOLODataset = YOLOWeightedDataset

    dataset_yaml = "./dataset_football/data.yaml"
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=1280,
        device=0,
        batch=1,
        conf=0.3
    )

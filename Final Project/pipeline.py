import cv2
from ultralytics import YOLO
import os
import yaml
from model import Multilabelresnet
import torch
import torchvision.transforms as transforms
from PIL import Image
from imutils.video import WebcamVideoStream, FPS
import time
from model_efficientNet import Multilabelefficient
from model import Multilabelresnet


def get_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]

    return class_names

def load_model_Object_Detection(model_path, device):
    model = YOLO(model_path).to(device)
    return model


def model_Object_Detection(model, input_img):

    results = model(input_img, stream=True, batch=1)

    object_info = []
    for result in results:
        bboxs = result.boxes.xyxy
        confs = result.boxes.conf
        class_ids = result.boxes.cls

        for i, box in enumerate(bboxs):
            x1, y1, x2, y2 = map(int, box)
            conf = confs[i]
            class_id = int(class_ids[i])
            object_info.append([x1,y1,x2,y2, conf,class_id])

    return object_info


def load_model_Classification(model_path, device):

    #device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = Multilabelresnet().to(device)
    model = Multilabelefficient().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    example_input = torch.randn(1, 3, 122, 224).to(device)
    model = torch.jit.trace(model, example_input)

    return model

def model_Classification(model, input_img, transform, device):

    img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs_number, outputs_color = model(img_tensor)
        number = torch.argmax(outputs_number, dim=1).item()
        color_id = (torch.sigmoid(outputs_color) > 0.6).long().item()

        color = "white" if color_id == 1 else "black"

    return number, color

def process_video(model_od_path, model_cls_path, input_video, device, transform, class_names, output_video):

    model_od = load_model_Object_Detection(model_od_path, device)
    model_cls = load_model_Classification(model_cls_path, device)

    # vid = WebcamVideoStream(src=input_video)
    # vid.start()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Không thể mở video")
        return

    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        #frame = vid.read()

        ret, frame = cap.read()

        if not ret:
            break

        od_info = model_Object_Detection(model_od, frame)
        for x1, y1, x2, y2, confidence, class_id in od_info:

            class_name = class_names[class_id]
            crop_img = frame[y1:y2, x1:x2]
            number, color = model_Classification(model_cls, crop_img, transform, device)

            if class_name == "player" and confidence > 0.7:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label1 = f"{class_name}: {confidence:.2f}" # for class name + conf
                label2 = f"{number} - {color}" # for number + color

                cv2.putText(frame, label1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 255, 0), 2)
                cv2.putText(frame, label2, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_name == "ball" and confidence > 0.35:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        width = 3840
        height = 1280
        resize_frame = cv2.resize(frame, (int(width // 2), int(height // 2)))

        cv2.imshow("Video Detection", resize_frame)
        out.write(frame)

        if cv2.waitKey(40) & 0xFF == ord("q"):
            break

    #vid.stop()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_yolo = 'runs/detect/train16/weights/best.pt'
    model_resnet = "best_model/best.pt"
    input_video_path = 'Datasets/football_test/Match_2031_5_0_test/Match_2031_5_0_test.mp4'
    output_video_path = 'output_video3.mp4'
    yaml_path = "dataset_football/data.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((122,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]
    process_video(model_yolo,model_resnet,input_video_path,device, transforms, class_names, output_video_path)

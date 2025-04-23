import comet_ml
from comet_ml import Experiment
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FootballDataset
from model_efficientNet import Multilabelefficient
from model_resnet import MultilabelResNet
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


experiment = Experiment(
    api_key="ciMxiKDwhZgTVTDdi4mrqXFKz",
    project_name="football-project",
    workspace="minhkhang1031"
)

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    batch_size = 16
    epochs = 50
    lr = 1e-3
    L2 = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = MultilabelResNet().to(device)
    model = Multilabelefficient().to(device)

    checkpoint_path = "last.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

    train_transform = transforms.Compose([
        transforms.Resize((122,224)),
        transforms.RandomAffine(
            degrees=(-10, 10), # xoay ảnh
            translate=(0.15, 0.15), # di chuyển ảnh theo chiều ngang - dọc
            scale=(0.85, 1.2), # phóng to - thu nhỏ
            shear=(-5, 5), # biến đổi theo dạng xéo
            interpolation=transforms.InterpolationMode.BILINEAR # nội suy => giữ nguyên chất lượng ảnh sau khi augmentation
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2), # tăng độ nét
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((122,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FootballDataset(csv_file="player_train.csv", root_dir="player_new", transform=train_transform)
    valid_dataset = FootballDataset(csv_file="player_val.csv", root_dir="player_new", transform=valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    criterion_number = nn.CrossEntropyLoss()
    criterion_color = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=L2)
    best_loss = float('inf')

    for epoch in range(epochs):
        #train
        model.train()
        running_loss = 0

        loop = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=True)
        for images, number_labels, color_labels in loop:
            images = images.to(device)
            number_labels = number_labels.to(device).long()
            color_labels = color_labels.to(device).float()

            optimizer.zero_grad()
            outputs_number, outputs_color = model(images)

            loss_number = criterion_number(outputs_number, number_labels)
            loss_color = criterion_color(outputs_color, color_labels.unsqueeze(1))
            loss = loss_number + loss_color

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_dataloader)

        #validation
        model.eval()
        valid_loss = 0
        all_preds_number, all_labels_number = [], []
        all_preds_color, all_labels_color = [], []
        with torch.no_grad():
            for images, number_labels, color_labels in val_dataloader:
                images = images.to(device)
                number_labels = number_labels.to(device).long()
                color_labels = color_labels.to(device).float()

                outputs_number, outputs_color = model(images)

                loss_number = criterion_number(outputs_number, number_labels)
                loss_color = criterion_color(outputs_color.squeeze(1), color_labels.float())
                valid_loss += (loss_number + loss_color).item()

                prediction_number = torch.argmax(outputs_number, dim=1).cpu().numpy()
                prediction_color = (torch.sigmoid(outputs_color) > 0.8).long().cpu().numpy()

                all_preds_number.extend(prediction_number)
                all_labels_number.extend(number_labels.cpu().tolist())

                all_preds_color.extend(prediction_color.flatten())
                all_labels_color.extend(color_labels.cpu().numpy().flatten())

        avg_valid_loss = valid_loss / len(val_dataloader)


        number_acc = accuracy_score(all_labels_number, all_preds_number)
        number_f1 = f1_score(all_labels_number, all_preds_number, average="macro")
        number_precision = precision_score(all_labels_number, all_preds_number, average="macro")
        number_recall = recall_score(all_labels_number, all_preds_number, average="macro")

        color_acc = accuracy_score(all_labels_color, all_preds_color)
        color_f1 = f1_score(all_labels_color, all_preds_color, average="binary")
        color_precision = precision_score(all_labels_color, all_preds_color, average="binary")
        color_recall = recall_score(all_labels_color, all_preds_color, average="binary")

        cm_number = confusion_matrix(all_labels_number, all_preds_number)
        cm_color = confusion_matrix(all_labels_color, all_preds_color)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}\n "
              f"Number - Acc: {number_acc:.2f}, Precision: {number_precision:.2f}, Recall: {number_recall:.2f}, F1: {number_f1:.2f}\n"
              f"Color  - Acc: {color_acc:.2f}, Precision: {color_precision:.2f}, Recall: {color_recall:.2f}, F1: {color_f1:.2f}")

        experiment.log_metrics({
            "train_loss": avg_loss,
            "valid_loss": avg_valid_loss,
            "number_acc": number_acc,
            "number_f1": number_f1,
            "number_precision": number_precision,
            "number_recall": number_recall,
            "color_acc": color_acc,
            "color_f1": color_f1,
            "color_precision": color_precision,
            "color_recall": color_recall,
        }, step=epoch)

        if avg_valid_loss < best_loss:

            best_loss = avg_valid_loss

            torch.save(model.state_dict(), "best.pt")

            plot_confusion_matrix(cm_number, "Confusion Matrix - Jersey Number",
                                  "resnet_both_invisible_and_partially/best_confusion_matrix_number.png")
            plot_confusion_matrix(cm_color, "Confusion Matrix - Jersey Color",
                                  "resnet_both_invisible_and_partially/best_confusion_matrix_color.png")

            experiment.log_image("best_confusion_matrix_number.png", name="Best_CM_JerseyNumber")
            experiment.log_image("best_confusion_matrix_color.png", name="Best_CM_JerseyColor")

        torch.save(model.state_dict(), "last.pt")


    comet_ml.end()


"""
📊 Classification report - Jersey Number:
              precision    recall  f1-score   support

           0      0.929     0.837     0.881       643
           1      0.981     0.998     0.990       521
           2      0.996     0.987     0.991       460
           3      0.982     0.995     0.989       440
           4      0.964     1.000     0.982       243
           5      0.982     0.992     0.987       611
           6      0.976     0.989     0.983       285
           7      0.958     0.987     0.972       371
           8      0.980     0.992     0.986       390
           9      0.970     0.998     0.984       457
          10      0.961     0.990     0.975       199
          11      0.978     0.977     0.977       643

    accuracy                          0.972      5263
   macro avg      0.971     0.978     0.975      5263
weighted avg      0.971     0.972     0.971      5263


📊 Classification report - Jersey Color:
              precision    recall  f1-score   support

        Dark      1.000     1.000     1.000      3485
       Light      1.000     0.999     1.000      1778

    accuracy                          1.000      5263
   macro avg      1.000     1.000     1.000      5263
weighted avg      1.000     1.000     1.000      5263
"""
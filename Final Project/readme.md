# REPORT FINAL PROJECT

The result for final project

## 1. Result for Object Detection
### Model

I choose model YOLO11n because its size is small, suitable for my GPU.

Parameters: 
 - imgsz: 1280
 - batch: 1
 - epochs: 70
 - conf: 0.3 (I choose small confidence to increase performance for the ball object)

Result: 
 - mAP50: 
 - mAP50-95:
 => Good performance for the player object. Although performance for the ball object quite lower but also it is increased significantly compared to init model.

 - Loss values in train and validation are decreasing over epochs 
 => No overfitting.

  - Precision and Recall achieve good values.
![image]() image result YOLO

Confusion matrix: 
 - Good performance with the player object.
 - Bad performance with the ball object. However, it is acceptable for small object.

![image]() image confusion maxtrix YOLO

### Dataset

- The dataset is imblanced data with a higher the player object compared to the ball object

![image]() image instance dataset YOLO

## 2. Result of Classification 
### Model

I choose Resnet50 for classifier task. However, its performance is not as good as expect. It was overfitting. After that, I decided changes to EfficientNet B0. The result fingure out performance of EfficientNet B0 better than Resnet50.

![image]() image loss value Resnet50
![image]() image loss value EfficientNet B0

Parameters: 
 - img size: (112,224) -> because image in dataset is rectangle
 - batch: 16
 - epochs: 50
 - learning rate: 1e-3
 - Data Augmentation: RandomAffine 
 -> degrees: -10 10
    translate: 0.15
    scale: +0.85 -1.2
    shear: -5 5
    interpolation: BILINEAR

 - increase resolution: RandomAdjustSharpness
 - optimizer: Adam (weight_decay = 1e-4)
 - Loss Function:
    + CrossEntropy for number jersey
    + BCEWithLogitsLoss for color jersey (binary classification)

Result:
 - Number jersey
    + Accuracy:
    + Precision:
    + Recall:
    + F1:
 - Color jersey
    + Accuracy
    + Precision:
    + Recall:
    + F1:

Confusion matrix: 



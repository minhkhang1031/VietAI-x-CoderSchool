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
 - mAP50: 0.788
 - mAP50-95: 0.657
 - Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
    all       1813      21325      0.948      0.722      0.788      0.657
  player       1813      18111      0.974      0.995      0.995      0.892
    ball       1712       3214      0.922      0.449      0.581      0.421
   
 => Good performance for the player object. Although performance for the ball object quite lower but also it is increased significantly compared to init model.

 - Loss values in both training and validation sets decrease over epochs. 
 => No overfitting.

  - Precision and Recall achieve good values.

  "The first 50 epochs"

![results](https://github.com/user-attachments/assets/9ecd1740-02e0-417d-893c-c905e4098397)

  "The next 20 epochs"
  
![results](https://github.com/user-attachments/assets/071966b0-9207-462f-8705-4d4b23258edb)

Confusion matrix: 
 - Good performance with the player object.
 - Bad performance with the ball object. However, it is acceptable for small object.

 ![confusion_matrix](https://github.com/user-attachments/assets/d26bf784-f87b-4eb5-a72d-15c3f517a1e7)


### Dataset

- The dataset is imblanced data with a higher the player object compared to the ball object
=> I changed to Focal Loss and used class weight to solve imblanced data problem.

![confusion_matrix](https://github.com/user-attachments/assets/f997aaa0-a2b0-4a37-935f-90748ab9fd2d)


## 2. Result of Classification 
### Model

I choose Resnet50 for classifier task. However, its performance is not as good as expect. It was overfitting. After that, I decided changes to EfficientNet B0. The result fingure out performance of EfficientNet B0 better than Resnet50.

"RESNET50"

![image](https://github.com/user-attachments/assets/8e2ea598-a3d5-4774-8b39-5d6baec4d700)

![image](https://github.com/user-attachments/assets/4b3ad8ed-adb3-4af6-aee6-ea33bf1ece66)

"EFFICIENTNETB0"

![image](https://github.com/user-attachments/assets/f20b64eb-70f2-43ae-831a-9f14ee8f04e1)

![image](https://github.com/user-attachments/assets/a46ab4b7-cb37-49e5-9882-60b4cb42fd0f)


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
 -- Number jersey
   
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

 -- Color jersey
   
                precision    recall  f1-score   support

        Dark      1.000     1.000     1.000      3485
       Light      1.000     0.999     1.000      1778

    accuracy                          1.000      5263
   macro avg      1.000     1.000     1.000      5263
weighted avg      1.000     1.000     1.000      5263

==> Good performance on the color task. However, performance on the number task is not good when the player is partially visible or too far from the camera.

Confusion matrix: 
 - The model classification has good performance.
  
 - Number jersey
 ![cm_number](https://github.com/user-attachments/assets/85fcbc47-a431-4826-9c08-b25da1e0d71c)

 - Color jersey
 ![cm_color](https://github.com/user-attachments/assets/0b7708d2-e364-4e17-9a25-a09e581eeaa8)


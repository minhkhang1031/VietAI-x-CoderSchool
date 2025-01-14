import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = x_train / 255
x_test = x_test / 255

y_train = y_train.flatten()
y_test = y_test.flatten()

#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = XGBClassifier()
model.fit(x_train, y_train)

# Predict on the test data
y_predict = model.predict(x_test)

# Print the classification report
print(classification_report(y_test, y_predict))

"""

RandomForestRegression
              precision    recall  f1-score   support

           0       0.54      0.56      0.55      1000
           1       0.52      0.54      0.53      1000
           2       0.38      0.33      0.35      1000
           3       0.33      0.28      0.30      1000
           4       0.39      0.38      0.39      1000
           5       0.43      0.40      0.41      1000
           6       0.47      0.57      0.52      1000
           7       0.51      0.45      0.48      1000
           8       0.58      0.61      0.59      1000
           9       0.47      0.52      0.50      1000

    accuracy                           0.47     10000
   macro avg       0.46      0.47      0.46     10000
weighted avg       0.46      0.47      0.46     10000


"""

"""
XGBoost
              precision    recall  f1-score   support

           0       0.61      0.60      0.61      1000
           1       0.66      0.64      0.65      1000
           2       0.43      0.40      0.41      1000
           3       0.38      0.37      0.37      1000
           4       0.45      0.45      0.45      1000
           5       0.47      0.47      0.47      1000
           6       0.55      0.64      0.59      1000
           7       0.60      0.55      0.58      1000
           8       0.65      0.68      0.66      1000
           9       0.58      0.59      0.58      1000

    accuracy                           0.54     10000
   macro avg       0.54      0.54      0.54     10000
weighted avg       0.54      0.54      0.54     10000
"""


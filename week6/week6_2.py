import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("./Week6_7/stroke_classification.csv")

#split data
x = data.drop(["stroke","pat_id"], axis = 1)
y = data["stroke"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=12)
# preprocess
#for numrical data
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#for ord binary data
gender = x_train["gender"].unique()

binary_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])

#transform

preprocess = ColumnTransformer(transformers=[
    ("num_transformer", num_transformer, ["age", "avg_glucose_level", "bmi"]),
    ("nom_transformer", nom_transformer, ["gender"]),
    ("binary_transformer", binary_transformer, ["hypertension", "heart_disease", "work_related_stress", "urban_residence", "smokes"])
])

# model

model = Pipeline(steps=[
    ("preporcessor", preprocess),
    ("model", RandomForestClassifier(random_state=12))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))

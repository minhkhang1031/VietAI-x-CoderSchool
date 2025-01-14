import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

#load data
data = pd.read_excel("./Week6_7/diabetes_data.xlsx")

#split data
x = data.drop(["Obesity", "DiabeticClass"], axis=1)

def target(tg):
    if tg["Obesity"] == "Yes" and tg["DiabeticClass"] == "Positive":
        return 3 # bi ca 2
    elif tg["Obesity"] == "Yes":
        return 1 # beo phi
    elif tg["DiabeticClass"] == "Positive":
        return 2 # tieu duong
    else:
        return 0 # khong bi gi

y = data.apply(lambda x: target(x), axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=12)

# preprocess => use ordinal transformer because data in dataset is binary
Gender = x_train["Gender"].unique()
ExcessUrination = x_train["ExcessUrination"].unique()
Polydipsia = x_train["Polydipsia"].unique()
WeightLossSudden = x_train["WeightLossSudden"].unique()
Fatigue = x_train["Fatigue"].unique()
Polyphagia = x_train["Polyphagia"].unique()
GenitalThrush = x_train["GenitalThrush"].unique()
BlurredVision = x_train["BlurredVision"].unique()
Itching = x_train["Itching"].unique()
Irritability = x_train["Irritability"].unique()
DelayHealing = x_train["DelayHealing"].unique()
PartialPsoriasis = x_train["PartialPsoriasis"].unique()
MuscleStiffness = x_train["MuscleStiffness"].unique()
Alopecia = x_train["Alopecia"].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[Gender, ExcessUrination,Polydipsia,WeightLossSudden,Fatigue,Polyphagia,GenitalThrush,BlurredVision,
                                     Itching,Irritability,DelayHealing,PartialPsoriasis,MuscleStiffness,Alopecia]))
])


preprocess = ColumnTransformer(transformers=[
    ("ord_transfomer", ord_transformer,
        ["Gender", "ExcessUrination", "Polydipsia", "WeightLossSudden", "Fatigue", "Polyphagia", "GenitalThrush", "BlurredVision",
         "Itching", "Irritability", "DelayHealing", "PartialPsoriasis", "MuscleStiffness", "Alopecia"])
])

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classicafication", RandomForestClassifier(random_state=12))
])

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


data1 = pd.read_csv("./Week6_7/csgo.csv")

data = data1.drop(["day", "month","year","date"], axis=1)

#split data

x = data.drop("points", axis=1)
y = data["points"]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=12)

# preprocess

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # Bổ sung handle_unknown
])

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocess = ColumnTransformer(transformers=[
    ("nom_transformer", nom_transformer, ["map", "result"]),
    ("num_transformer", num_transformer, ["wait_time_s", "team_a_rounds", "team_b_rounds", "ping", "kills", "assists", "deaths", "mvps", "hs_percent"])
])

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocess),
    ("model", RandomForestRegressor(random_state=12))
])

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_predict)
print(f"MAE: {mae}")
r2 = r2_score(y_test, y_predict)
print(f"R-squared: {r2}")


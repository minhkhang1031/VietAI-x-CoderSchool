import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("Time-series-datasets/electricity.csv")

# data["Time"] = data["Time"].to_timestamp()
# data["Date"] = data["Date"].to_timestamp()

def create_ts(data, windows_slide):
    i = 1
    while i < windows_slide:
        data[f'Demand_{i}'] = data["Demand"].shift(-1)
        i += 1
    data["Target"] = data["Demand"].shift(-1)
    data = data.dropna(axis=0)
    return data

window_slide = 5
ratio = 0.8
sample = len(data)

data = create_ts(data, window_slide)
x = data.drop(["Time","Date","Target"], axis = 1)
y = data["Target"]


x_train = x[:int(sample * ratio)]
x_test = x[int(sample * ratio):]
y_train = y[:int(sample * ratio):]
y_test = y[int(sample * ratio):]

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(f'MAE: {mean_absolute_error(y_predict, y_test)}')
print(f'R2: {r2_score(y_predict, y_test)}')


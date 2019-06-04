import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pickle

dataset = pd.read_csv("FuelConsumptionCo2.csv")
dataset.head()

df = dataset[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
df.head()

def graph(x, y, x_label, y_label):
    plt.scatter(x, y, color="blue")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


graph(df.ENGINESIZE, df.CO2EMISSIONS, "Engine Size", "CO2 Emissions")
graph(df.CYLINDERS, df.CO2EMISSIONS, "Cylinders", "CO2 Emissions")
graph(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS, "Fuel Consumption", "CO2 Emissions")


n = np.random.rand(len(df)) < 0.8
train = df[n]
test = df[~n]

x_train, y_train = train[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]], \
                   train[["CO2EMISSIONS"]]
x_test, y_test = test[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]], \
                 test[["CO2EMISSIONS"]]


model = linear_model.LinearRegression()
model.fit(x_train, y_train)

saved_model = pickle.dumps(model)

y_prediction_values = model.predict(x_test)
residual_sum_squares = np.mean((y_prediction_values - y_test) ** 2)
print(residual_sum_squares)

variance_score = model.score(x_test, y_test)
print(variance_score)

lst_input = [[]]

f = open("out.txt", "r+")
f1 = open("out3.txt", "r+")

def get_user_data(file_name):
    global lst_input
    f_read = file_name.readlines()

    for element in f_read:
        lst_input[0].append(float(element))

def predict(file_name, model_name):
    loaded_model = pickle.loads(model_name)

    prediction_string = str(loaded_model.predict(np.array(lst_input)))
    prediction = prediction_string[2:-2]

    file_name.write(prediction)
    file_name.close()

get_user_data(f)
predict(f1, saved_model)


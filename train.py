from sklearn.model_selection import train_test_split
from model import NNModel
from pandas import read_csv
import numpy as np


dataset = read_csv("dataset/spambase_csv.csv")

model = NNModel("he_uniform")

y = dataset["class"]


del dataset["class"]

X = dataset

print(len(list(X)))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

model.train(X_train, np.asarray(y_train.astype("float32")).reshape((-1, 1)))

print(model.model.metrics_names)
print(
    model.evaluate(
        X_test, np.asarray(y_test.astype("float32")).reshape((-1, 1))
    )
)

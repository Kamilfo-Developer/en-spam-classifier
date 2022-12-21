from keras.losses import BinaryCrossentropy
from keras.layers import Dense, InputLayer
from keras import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from pandas import DataFrame
from random import randint


class NNModel:
    def __init__(self, *args, **kwargs):
        self.model = NNModel._get_model(*args, **kwargs)

    @staticmethod
    def _get_model(weight_initializer: str) -> Sequential:
        model = Sequential()

        model.add(InputLayer(57))

        model.add(
            Dense(
                100,
                "sigmoid",
                weight_initializer,
            )
        )

        model.add(Dense(75, "sigmoid", weight_initializer))

        model.add(Dense(50, "sigmoid", weight_initializer))

        model.add(Dense(1, "sigmoid", weight_initializer))

        optimizer = RMSprop()

        loss = BinaryCrossentropy()

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy", "binary_accuracy"],
        )

        return model

    def train(self, X: DataFrame, y: DataFrame):
        self.model.fit(X, y, epochs=100)

    def predict(self, X: DataFrame):
        return self.model.predict(X)

    def evaluate(self, X: DataFrame, y: DataFrame):
        return self.model.evaluate(X, y)

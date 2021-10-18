import math
import os
import cv2 as cv

import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn import model_selection
from tensorflow.python.keras.models import Sequential
from keras import models
from keras.optimizers import SGD


def load_data():
    data = pd.read_csv("image_data.csv")
    y = data["y"].to_numpy()
    x = data.iloc[:, 1:-1].to_numpy()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.01)

    x_train = x_train.reshape((x_train.shape[0], int(math.sqrt(x_train.shape[1])), int(math.sqrt(x_train.shape[1])), 1))
    x_test = x_test.reshape((x_test.shape[0], int(math.sqrt(x_test.shape[1])), int(math.sqrt(x_test.shape[1])), 1))

    y_test = np_utils.to_categorical(y_test)[:, 1:]
    y_train = np_utils.to_categorical(y_train)[:, 1:]

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    return x_train, x_test, y_train, y_test


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def test_model(x, y, model):
    _, acc = model.evaluate(x, y, verbose=0)
    print(f"Accuracy: {acc * 100}")


def predict(img):
    size = img.shape[0]
    converted = np.array([img.reshape((size, size, 1))]) / 255
    predictions = recognizer.predict(converted)
    for i, prediction in enumerate(predictions[0]):
        if prediction > 0.7:
            return 0 if i == 9 else i + 1
    return 0


if not os.path.isdir("model"):
    print("training")
    x_train, x_test, y_train, y_test = load_data()
    model = create_model()
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test), verbose=0)
    model.save("model")

recognizer = models.load_model("model")















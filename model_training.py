
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

train_prepared = joblib.load("train_prepared.pkl")
train_labels = joblib.load("train_labels.pkl")

model = keras.models.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=train_prepared.shape[1:]),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1) 
])

model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["mae"])

history = model.fit(train_prepared, train_labels, epochs=50, validation_split=0.2)

model.save("model.h5")

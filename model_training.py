import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 전처리된 데이터 불러오기
fires_prepared = joblib.load("train_prepared.pkl")
fires_labels = joblib.load("train_labels.pkl")
fires_test_prepared = joblib.load("test_prepared.pkl")
fires_test_labels = joblib.load("test_labels.pkl")

X_train, Xvalid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["mae"])
history = model.fit(X_train, y_train, epochs=50, validation_data=(Xvalid, y_valid))

model.save("fires_model.keras")
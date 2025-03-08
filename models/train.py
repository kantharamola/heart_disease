import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_rnn_model import build_model

# Load dataset
df = pd.read_csv("data/processed_features.csv")
X = df.iloc[:, 1:-1].values  # Exclude patient_id and label
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build model
model = build_model(input_shape=(X_train.shape[1], 1))

# TensorBoard setup
log_dir = "logs/fit"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Save model
model.save("models/heart_disease_model.h5")
print("Training complete. Model saved.")

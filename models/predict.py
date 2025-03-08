import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("models/heart_disease_model.h5")

def predict(features):
    features = np.array(features).reshape(1, len(features), 1)  # Reshape for CNN input
    prediction = model.predict(features)[0][0]
    return "Heart Disease Detected" if prediction > 0.5 else "No Heart Disease"

# Example Usage
df = pd.read_csv("data/processed_features.csv")
sample = df.iloc[0, 1:-1].values  # First sample excluding patient_id and label
print(predict(sample))

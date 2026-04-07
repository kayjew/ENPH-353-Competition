import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(36, activation='softmax')
    ])
    return model

data = np.load('../models/clue_weights.npz')
weights = [data[f'w{i}'] for i in range(len(data))]
print(f"Loaded {len(weights)} weight arrays")

model = build_model()
model.set_weights(weights)
model.save('../models/clue_reader_local.h5')
print("Saved to models/clue_reader_local.h5")
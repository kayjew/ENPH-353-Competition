#!/usr/bin/env python3
"""
Trains NN to recognize chars from boards
"""

import os
import sys
import random
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from clue_detection.model_utils import BoardProcessor, char_to_int, CHARS


def load_dataset(data_path):
    """
    Looks for data
    """
    processor = BoardProcessor()
    X, y = [], []

    if not os.path.exists(data_path):
        print(f"[ERROR] Path not found: {data_path}")
        return np.array([]), np.array([])

    files = [f for f in os.listdir(data_path) if f.endswith('.png')]
    if not files:
        print(f"[ERROR] No PNG files found in: {data_path}")
        return np.array([]), np.array([])

    random.shuffle(files)
    print(f"[INFO] Processing {len(files)} images...")

    for fname in files:

        parts = fname.replace('.png', '').split('_')
        if len(parts) < 2:
            continue

        type_label = parts[0]
        val_label  = parts[1]

        img = cv2.imread(os.path.join(data_path, fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        type_imgs, val_imgs = processor.segment_both(gray)

        if len(type_imgs) == len(type_label):
            for char_img, char_txt in zip(type_imgs, type_label):
                if char_txt in char_to_int:
                    X.append(char_img)
                    y.append(char_to_int[char_txt])

        if len(val_imgs) == len(val_label):
            for char_img, char_txt in zip(val_imgs, val_label):
                if char_txt in char_to_int:
                    X.append(char_img)
                    y.append(char_to_int[char_txt])

    if len(X) == 0:
        print("[ERROR] No characters extracted. Check ROI coordinates and image format.")
        return np.array([]), np.array([])

    X = np.array(X).reshape(-1, 32, 32, 1) / 255.0
    y = np.array(y)
    print(f"[INFO] Extraction complete. Total character samples: {len(X)}")
    return X, y


def build_model(num_classes=36):
    """CNN architecture for 32x32 greyscale"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def run_training(data_path, save_path='clue_reader_model.h5'):
    X, y = load_dataset(data_path)
    if len(X) == 0:
        print("[ERROR] No data loaded. Aborting.")
        return None

    # 70% train 15% val 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = build_model(num_classes=len(CHARS))

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print("[INFO] Starting training...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Evaluation
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Final Test Accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap='Blues',
                xticklabels=CHARS, yticklabels=CHARS)
    plt.title("Character Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path.replace('.h5', '_confusion.png'))
    plt.show()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clue board character recognizer")
    parser.add_argument('--data', type=str, default='./dataset',
                        help='Path to directory containing training PNG images')
    parser.add_argument('--out',  type=str, default='../models/clue_reader_model.h5',
                        help='Where to save the trained model')
    args = parser.parse_args()

    run_training(args.data, args.out)
from shared import PROCESSED_DATA_DIR, MODEL_DIR, CATEGORIES, DRAWING_COUNT, VALIDATION_FRACTION, IMAGE_SIZE
import random
from datetime import datetime
import os.path as path
import numpy as np
import tensorflow as tf
from keras import models, layers, losses, callbacks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
SHUFFLE_SIZE = 10000

date = datetime.now().strftime(r"%Y%m%d_%H%M")

def load_data():
    data = []
    offsets = [0]

    for category in CATEGORIES:
        print(f"Loading {category} data")

        category_file_path = path.join(PROCESSED_DATA_DIR, f"{category}.npy")
        category_data = np.load(file=category_file_path, mmap_mode="r")
        data.append(category_data[:DRAWING_COUNT])
        offsets.append(offsets[-1] + DRAWING_COUNT)

    return data, offsets

def map_data(example):
    features = tf.io.parse_single_example(example, {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string)
    })
    image = tf.io.decode_raw(features["image"], tf.uint8)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 1))
    image = tf.cast(image, tf.float32) / 255

    return image, features["label"]

def build_dataset(dataset_file_pattern, shuffle=True):
    file_list = tf.data.Dataset.list_files(dataset_file_pattern, shuffle=shuffle)
    dataset = file_list.interleave(tf.data.TFRecordDataset, cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_SIZE)

    dataset = dataset.map(map_data, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

def visualize_data(training_dataset):
    images, labels = next(iter(training_dataset.take(1)))

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(CATEGORIES[labels[i]])
    
    plt.show()

def create_model():
    model = models.Sequential()
    model.add(layers.Input((IMAGE_SIZE, IMAGE_SIZE, 1)))

    return model

def augment_model(model):
    model.add(layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=0, interpolation="nearest"))
    model.add(layers.RandomZoom(height_factor=(0, 0.2), fill_mode="constant", fill_value=0, interpolation="nearest"))
    model.add(layers.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE))

def complete_model(model):
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(len(CATEGORIES)))
    model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

def train_model(model, training_dataset, validation_dataset):
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    training_info = model.fit(training_dataset, epochs=10, validation_data=validation_dataset, callbacks=[early_stop])
    model.save(path.join(MODEL_DIR, f"{date}.keras"))

    return training_info

def visualize_augmentation(model, training_dataset):
    images, labels = next(iter(training_dataset.take(1)))
    augmentation_y = model.call(images, training=True)

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        original_axis = axes[0, i]
        augmented_axis = axes[1, i]

        original_axis.imshow(images[i])
        augmented_axis.imshow(augmentation_y[i])
    
    plt.show()

def visualize_results(model, validation_dataset, validation_index_data, training_info):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_axis = axes[0]
    accuracy_axis = axes[1]

    y_true = get_category(validation_index_data)
    y_prediction = model.predict(validation_dataset)
    y_prediction = np.argmax(y_prediction, axis=1)
    matrix = confusion_matrix(y_true, y_prediction)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CATEGORIES)
    matrix_display.plot(ax=cm_axis, colorbar=False)
    cm_axis.set_title("Confusion Matrix")
    cm_axis.set_xticklabels(cm_axis.get_xticklabels(), rotation=90)

    accuracy_axis.plot(training_info.history["accuracy"], label="accuracy")
    accuracy_axis.plot(training_info.history["val_accuracy"], label="val_accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim([0.5, 1])
    accuracy_axis.legend(loc="lower right")
    accuracy_axis.set_title("Accuracy")

    plt.show()

data, offsets = load_data()
training_dataset = build_dataset(f"{PROCESSED_DATA_DIR}/*/training*")
validation_dataset = build_dataset(f"{PROCESSED_DATA_DIR}/*/validation*", shuffle=False)
visualize_data(training_dataset)
model = create_model()
augment_model(model)
visualize_augmentation(model, training_dataset)
complete_model(model)
training_info = train_model(model, training_dataset, validation_dataset)
visualize_results(model, validation_dataset, validation_index_data, training_info)

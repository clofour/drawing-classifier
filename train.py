from shared import PROCESSED_DATA_DIR, MODEL_DIR, CATEGORIES, IMAGE_SIZE
import random
from datetime import datetime
import os.path as path
import numpy as np
import tensorflow as tf
from keras import models, layers, losses, callbacks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DRAWING_COUNT = 100000
VALIDATION_FRACTION = 0.2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
SHUFFLE_SIZE = 10000

date = datetime.now().strftime(r"%Y%m%d_%H%M")

data = []
offsets = [0]

for category in CATEGORIES:
    print(f"Loading {category} data")

    category_file_path = path.join(PROCESSED_DATA_DIR, f"{category}.npy")
    category_data = np.load(file=category_file_path, mmap_mode="r")
    data.append(category_data[:DRAWING_COUNT])
    offsets.append(offsets[-1] + DRAWING_COUNT)

shuffled_indexes = np.random.permutation(offsets[-1])
validation_drawing_count = int(len(CATEGORIES) * DRAWING_COUNT * VALIDATION_FRACTION)
validation_indexes = shuffled_indexes[validation_drawing_count:]
training_indexes = shuffled_indexes[:validation_drawing_count]

def process(image):
    image = np.array(image, np.float32)
    image = image / 255
    image = image.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

    return image

def lookup(global_index):
    global_index = int(global_index)

    category_id = np.searchsorted(offsets[1:], global_index)
    category_start = offsets[category_id]
    local_index = global_index - category_start

    image = data[category_id][local_index]
    processed_image = process(image)

    return (processed_image, category_id)

def tf_lookup(global_index):
    image, label = tf.numpy_function(lookup, [global_index], [tf.float32, tf.int32])
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))
    label.set_shape(())

    return image, label

def build_dataset(index_data, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(index_data)

    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_SIZE)

    dataset = dataset.map(tf_lookup, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

training_dataset = build_dataset(training_indexes)
validation_dataset = build_dataset(validation_indexes, shuffle=False)


def visualize_data(data):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        chosen_category = random.choice(CATEGORIES)
        bitmap = data[chosen_category][i]
        image = bitmap.reshape(IMAGE_SIZE, IMAGE_SIZE)

        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
        plt.xlabel(chosen_category)
    
    plt.show()

def process_data(data):
    images = []
    labels = []

    for label, category in enumerate(CATEGORIES):
        category_data = data[category]
        images.append(category_data[:DRAWING_COUNT])
        labels.append(np.full(DRAWING_COUNT, label))

    images = np.concatenate(images)
    labels = np.concatenate(labels)

    images = np.array(images)
    labels = np.array(labels)

    images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    images = images / 255.0

    permutation = np.random.permutation(len(images))
    images = images[permutation]
    labels = labels[permutation]

    data_split_index = int(len(images) * VALIDATION_FRACTION)
    x_training = images[data_split_index:]
    y_training = labels[data_split_index:]
    x_validation = images[:data_split_index]
    y_validation = labels[:data_split_index]

    return x_training, y_training, x_validation, y_validation

def create_model():
    model = models.Sequential()
    model.add(layers.Input((IMAGE_SIZE, IMAGE_SIZE, 1)))

    return model

def augment_model(model):
    model.add(layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=0, interpolation="nearest"))
    model.add(layers.RandomZoom(height_factor=(0, 0.2), fill_mode="constant", fill_value=0, interpolation="nearest"))
    model.add(layers.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE))

def complete_model(model):
    model.add(layers.Conv2D(32, (3, 3)), padding="same")
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3)), padding="same")
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3)), padding="same")
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(len(CATEGORIES)))
    model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

def train_model(model, x_training, y_training, x_validation, y_validation):
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    training_info = model.fit(x=x_training, y=y_training, epochs=10, validation_data=(x_validation, y_validation), callbacks=[early_stop])
    model.save(path.join(MODEL_DIR, f"{date}.keras"))

    return training_info

def visualize_augmentation(model, x_training, y_training):
    sample_x = x_training[:5]
    augmentation_y = model.call(sample_x, training=True)

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        original_axis = axes[0, i]
        augmented_axis = axes[1, i]

        original_axis.imshow(sample_x[i].squeeze())
        augmented_axis.imshow(augmentation_y[i].numpy().squeeze())
    
    plt.show()
        

def visualize_results(model, x_training, y_training, x_validation, y_validation, training_info):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_axis = axes[0]
    accuracy_axis = axes[1]

    y_prediction = model.predict(x_validation)
    y_prediction = np.argmax(y_prediction, axis=1)
    matrix = confusion_matrix(y_validation, y_prediction)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CATEGORIES)
    matrix_display.plot(ax=cm_axis, colorbar=False)
    cm_axis.set_title("Confusion Matrix")

    accuracy_axis.plot(training_info.history["accuracy"], label="accuracy")
    accuracy_axis.plot(training_info.history["val_accuracy"], label="val_accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim([0.5, 1])
    accuracy_axis.legend(loc="lower right")
    accuracy_axis.set_title("Accuracy")

    plt.show()

data = load_data()
visualize_data(data)
x_training, y_training, x_validation, y_validation = process_data(data)
model = create_model()
augment_model(model)
visualize_augmentation(model, x_training, y_training)
complete_model(model)
training_info = train_model(model, x_training, y_training, x_validation, y_validation)
visualize_results(model, x_training, y_training, x_validation, y_validation, training_info)

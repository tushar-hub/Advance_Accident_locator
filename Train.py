from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow import keras
import tensorflow as tf
import numpy as np

print("Importing Done...")


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get("accuracy") > 0.985:
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


callback = myCallback()


def build(h, w, d, classes=1):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same", input_shape=(h, w, d)),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.MaxPool2D((2, 2)),

        keras.layers.Conv2D(32, (3, 3),padding="same", activation="relu"),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.MaxPool2D((2, 2)),

        keras.layers.Conv2D(32, (3, 3),padding="same", activation="relu"),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.MaxPool2D((2, 2)),

        # keras.layers.Conv2D(32, (5, 5), activation="relu"),
        # keras.layers.BatchNormalization(axis=-1),
        # keras.layers.MaxPool2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        # keras.layers.Dense(16, activation="relu"),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


train_path = "Datasets/train/"
validation_path = "Datasets/validation/"
BATCH_SIZE = 64
NUM_EPOCH = 100
EPOCH_PER_STEP = 8
HEIGHT = 256
WIDTH = 256
DEPTH = 3
classes = 2
learn = 0.01

print("Data Generating...")
train_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=30,
                                   zoom_range=0.15, width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.15,
                                   horizontal_flip=True,
                                   fill_mode="nearest")
vaildation_datagen = ImageDataGenerator(rescale=1 / 255)

print("Data loading...")
train_generator = train_datagen.flow_from_directory(
    'Datasets/train',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='binary')
validation_generator = vaildation_datagen.flow_from_directory(
    'Datasets/validation',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='binary')

print("Model Building...")
model = build(HEIGHT, WIDTH, DEPTH, classes)
print("Compling.....")
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=["accuracy"])

print("Traning Started...")
history = model.fit(
    train_generator,
    steps_per_epoch=EPOCH_PER_STEP,
    epochs=NUM_EPOCH,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=EPOCH_PER_STEP,
    callbacks=[callback])

print("Saving Model...")
model.save("model1.h5")

print("Program Done!")

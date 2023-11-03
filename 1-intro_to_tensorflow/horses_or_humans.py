from __future__ import annotations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.utils import img_to_array, load_img
from unzip_and_download import download_and_unzip

FILE_URLS = [
    "https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip",
    "https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip"
]

# download_and_unzip(file_urls=FILE_URLS)
train_dir = os.path.join("./horse-or-human")
validation_dir = os.path.join("./validation-horse-or-human")

train_horse_dir = os.path.join('./horse-or-human/horses')
train_human_dir = os.path.join('./horse-or-human/humans')
validation_horse_dir = os.path.join('./validation-horse-or-human/horses')
validation_human_dir = os.path.join('./validation-horse-or-human/humans')


def generate_model() -> tf.keras.Model.Sequential:
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


def get_generators() -> tuple[ImageDataGenerator, ImageDataGenerator]:
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        class_mode='binary')  # binary labels for binary_crossentropy loss

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        class_mode='binary')

    return train_generator, validation_generator


def train(model, data_generator, validation_generator, epochs):
    history = model.fit(
        data_generator,
        steps_per_epoch=8,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=8)
    return history


def visualise_intermediate_representations(model) -> None:
    horse_img_files = [os.path.join(train_horse_dir, f) for f in os.listdir(train_horse_dir)]
    human_img_files = [os.path.join(train_human_dir, f) for f in os.listdir(train_human_dir)]
    img_path = random.choice(horse_img_files + human_img_files)  # cat lists and get random image
    img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (300, 300, 3)
    x = x.reshape((1,) + x.shape) / 255  # Numpy array with shape (1, 300, 300, 3)

    layer_outputs = [layer.output for layer in model.layers[1:]]
    extractor = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    successive_feature_maps = extractor.predict(x)

    layer_names = [layer.name for layer in model.layers[1:]]

    for layer_name, feature_maps in zip(layer_names, successive_feature_maps):
        if len(feature_maps.shape) == 4:  # only conv and MP - not FC layers
            n_features = feature_maps.shape[-1]  # number of features in feature map
            size = feature_maps.shape[1]  # feature map shape (1, size, size, n_features)
            display_grid = np.zeros((size, size * n_features))  # We will tile our images in this matrix
            # e.g. if size = 150 and there are 32 filters, we'll get 150x150*32
            for i in range(n_features):  # for every feature map
                # Postprocess the feature to make it visually palatable
                x = feature_maps[0, :, :, i]  # 0th index, all rows and columns. ith feature
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into a big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x

            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()


if __name__ == '__main__':
    model = generate_model()
    train_generator, validation_generator = get_generators()
    history = train(model, train_generator, validation_generator, epochs=2)
    visualise_intermediate_representations(model)

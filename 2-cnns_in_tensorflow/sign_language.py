import csv
import numpy as np
import matplotlib.pyplot as plt
import string
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img

train_csv_file = 'sign_language/sign_mnist_train/sign_mnist_train.csv'
test_csv_file = 'sign_language/sign_mnist_test/sign_mnist_test.csv'


def parse_data_from_input(filename):
    labels = []
    images = []

    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        for line in reader:
            labels.append(line[0])
            images.append(line[1:])

    labels = np.array(labels).astype('float64')
    images = np.array(images).astype('float64').reshape(-1, 28, 28)
    return images, labels


def plot_categories(training_images, training_labels):
    fig, axes = plt.subplots(1, 10, figsize=(16, 15))
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    for k in range(10):
        img = training_images[k]
        img = np.expand_dims(img, axis=-1)
        img = array_to_img(img)
        ax = axes[k]
        ax.imshow(img, cmap="Greys_r")
        ax.set_title(f"{letters[int(training_labels[k])]}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.legacy.Adamax(learning_rate=0.0025), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_val_generators(training_images, training_labels, validation_images, validation_labels):
    training_images = np.expand_dims(training_images, -1)
    validation_images = np.expand_dims(validation_images, -1)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1
    )
    train_generator = train_datagen.flow(x=training_images,
                                         y=training_labels,
                                         batch_size=32)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow(x=validation_images,
                                                   y=validation_labels,
                                                   batch_size=32)

    return train_generator, validation_generator


if __name__ == '__main__':
    training_images, training_labels = parse_data_from_input(train_csv_file)
    test_images, test_labels = parse_data_from_input(test_csv_file)
    # plot_categories(training_images, training_labels)
    train_generator, validation_generator = train_val_generators(training_images, training_labels, test_images, test_labels)
    model = create_model()
    history = model.fit(train_generator,
                        epochs=15,
                        validation_data=validation_generator,
                        steps_per_epoch=200)

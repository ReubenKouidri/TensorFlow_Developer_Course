import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') and logs.get('accuracy') >= 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


def plot(images, labels, index):
    # Print the label and image
    print(f'LABEL: {labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {images[index]}')
    plt.imshow(train_imgs[index])
    plt.show()


def generate_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer=tf.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def reshape_and_normalise(images):
    images = images.reshape(-1, 28, 28, 1)
    images = images / np.max(images)
    return images


def generate_data() -> tuple[tuple, tuple]:
    fmnist = tf.keras.datasets.fashion_mnist
    (training_imgs, training_labels), (test_images, test_labels) = fmnist.load_data()

    training_imgs = reshape_and_normalise(training_imgs)
    test_images = reshape_and_normalise(test_images)

    return (training_imgs, training_labels), (test_images, test_labels)


if __name__ == '__main__':
    model = generate_model()
    train_data, test_data = generate_data()
    (train_imgs, train_tgts), (test_imgs, test_tgts) = train_data, test_data

    callbacks = MyCallback()
    model.fit(train_imgs, train_tgts, epochs=5, callbacks=[callbacks])
    model.evaluate(test_imgs, test_tgts)  # Evaluate the model on unseen data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= 0.90:
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True


def plot(images, labels, index):
    # Print the label and image
    print(f'LABEL: {labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {images[index]}')
    plt.imshow(train_imgs[index])
    plt.show()


def generate_inputs():
    inputs = np.array([[1.0, 3.0, 4.0, 2.0]])  # Declare sample inputs
    inputs = tf.convert_to_tensor(inputs)  # convert to a tensor
    return inputs


def generate_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    return model


def generate_data() -> tuple[tuple, tuple]:
    fmnist = tf.keras.datasets.fashion_mnist
    (training_imgs, training_labels), (test_images, test_labels) = fmnist.load_data()

    training_imgs = training_imgs / 255.0
    test_images = test_images / 255.0
    return (training_imgs, training_labels), (test_images, test_labels)


if __name__ == '__main__':
    model = generate_model()
    train_data, test_data = generate_data()
    (train_imgs, train_tgts), (test_imgs, test_tgts) = train_data, test_data
    inputs = generate_inputs()

    model.compile(optimizer=tf.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_imgs, train_tgts, epochs=5)  # train the model
    model.evaluate(test_imgs, test_tgts)  # Evaluate the model on unseen data

    callbacks = MyCallback()
    model.fit(train_imgs, train_tgts, epochs=5, callbacks=[callbacks])

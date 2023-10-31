import matplotlib.pyplot as plt
import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


def generate_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    return model


def generate_dataset() -> tf.data:
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), _ = mnist.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    return training_images, training_labels


def main():
    imgs, tgts = generate_dataset()
    callbacks = MyCallback()
    model = generate_model()

    optimizer = tf.keras.optimizers.legacy.Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(imgs, tgts, epochs=10, callbacks=[callbacks])
    return history


if __name__ == "__main__":
    hist = main()
    # plt.plot(hist.history['accuracy'])
    # plt.show()

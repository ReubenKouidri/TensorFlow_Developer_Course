import fashion_mnist as fashion
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_model(model, test_images):
    first_image = 0
    second_image = 23
    third_image = 28
    convolution_number = 1

    f, axarr = plt.subplots(3, 4)
    # layer_outputs = output from each of the 7 layers defined below
    # e.g. layer_outputs[0] = Tensor (None, 26, 26, 32) with weights optimised
    # e.g. layer_outputs[1] = Tensor (None, 13, 13, 64) etc...
    layer_outputs = [layer.output for layer in model.layers]
    # model.input is a single image (None, 28, 28, 1)
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    #
    for x in range(0, 4):
        # returns num_filters in layer tensors * output size (e.g. 26x26). So overall f1.size = (1, 26, 26, 32)
        f1 = activation_model.predict(test_images[first_image].reshape(1, 28, 28, 1))[x]
        axarr[0, x].imshow(f1[0, :, :, convolution_number], cmap='inferno')
        axarr[0, x].grid(False)

        f2 = activation_model.predict(test_images[second_image].reshape(1, 28, 28, 1))[x]
        axarr[1, x].imshow(f2[0, :, :, convolution_number], cmap='inferno')
        axarr[1, x].grid(False)

        f3 = activation_model.predict(test_images[third_image].reshape(1, 28, 28, 1))[x]
        axarr[2, x].imshow(f3[0, :, :, convolution_number], cmap='inferno')
        axarr[2, x].grid(False)
        plt.show()


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= 0.92:
            print("\nReached 92% accuracy so cancelling training!")
            self.model.stop_training = True


def generate_conv_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model


def main():
    train_data, test_data = fashion.generate_data()
    model = generate_conv_model()

    optimizer = tf.keras.optimizers.legacy.Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data[0], train_data[1], epochs=1, callbacks=[MyCallback()])
    model.evaluate(test_data[0], test_data[1])

    plot_model(model, test_data[0])


if __name__ == "__main__":
    main()

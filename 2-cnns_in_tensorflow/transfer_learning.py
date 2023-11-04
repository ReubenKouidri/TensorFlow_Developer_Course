import os
import tensorflow as tf
from keras import layers
from keras import Model
import keras.optimizers.legacy as optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights.h5'
base_dir = os.path.join('./cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


def train_val_generators(training_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(directory=training_dir,
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                  batch_size=32,
                                                                  class_mode='binary',
                                                                  target_size=(150, 150))
    return train_generator, validation_generator


def transfered_model():
    base_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None
    )
    base_model.load_weights(local_weights_file)
    for layer in base_model.layers:
        layer.trainable = False

    last_layer_name = 'mixed7'
    last_layer = base_model.get_layer(last_layer_name)
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    combined_model = Model(inputs=base_model.input, outputs=x)
    combined_model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
    return combined_model


if __name__ == '__main__':
    train_generator, validation_generator = train_val_generators(train_dir, validation_dir)
    model = transfered_model()
    model.fit(train_generator,
              epochs=100,
              validation_data=validation_generator,
              callbacks=[MyCallback()])
    model.save('transfer_learning.h5')

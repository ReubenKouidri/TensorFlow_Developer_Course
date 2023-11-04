import shutil
from utils.unzip_and_download import download_and_unzip
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as random
import tensorflow as tf
from keras.optimizers.legacy import Adamax
from keras.preprocessing.image import ImageDataGenerator

random.seed(984)
DOWNLOAD = True
DOWNLOAD_FULL = True
PLOT_IMAGES = False
SMALL_DATA_PATH = ["https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"]
FULL_DATA_PATH = [
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"]


def plot_images(cat_dir, dog_dir, num_images=16):
    nrows = int(num_images ** 0.5)
    ncols = int(num_images / nrows)
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    train_cat_fnames = os.listdir(cat_dir)
    train_dog_fnames = os.listdir(dog_dir)
    num_cat_pics = int(num_images / 2)
    num_dog_pics = int(num_images - num_cat_pics)

    next_cat_pix = [os.path.join(cat_dir, fname) for fname in
                    [random.choice(train_cat_fnames) for _ in range(num_cat_pics)]
                    ]
    next_dog_pix = [os.path.join(dog_dir, fname) for fname in
                    [random.choice(train_dog_fnames) for _ in range(num_dog_pics)]
                    ]

    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()


def generate_model():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
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
    model.compile(optimizer=Adamax(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_generators(train_dir, validation_dir):
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
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=100,
        class_mode='binary'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=100,
        class_mode='binary'
    )
    return train_generator, validation_generator


def train(model, train_generator, validation_generator, epochs=15):
    hist = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=1
    )
    return hist


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    x = range(len(acc))  # Get number of epochs
    plt.plot(x, acc)
    plt.plot(x, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(x, loss)
    plt.plot(x, val_loss)
    plt.show()


def split_data(source_dir, training_dir, validation_dir, split_size):
    file_paths = [
        os.path.join(source_dir, file)
        for file in os.listdir(source_dir)
        if os.path.getsize(os.path.join(source_dir, file)) != 0
    ]
    random.shuffle(file_paths)
    train_paths = file_paths[:int(split_size * len(file_paths))]
    validation_paths = file_paths[int(split_size * len(file_paths)):]

    for path in train_paths:
        shutil.copyfile(path, os.path.join(training_dir, os.path.basename(path)))
    for path in validation_paths:
        shutil.copyfile(path, os.path.join(validation_dir, os.path.basename(path)))


def make_directory_structure():
    base_dest_dir = './cats_vs_dogs/'
    source_dir = './cats_vs_dogs/PetImages/'
    cat_source_dir = os.path.join(source_dir, 'Cat')
    dog_source_dir = os.path.join(source_dir, 'Dog')

    train_dir = os.path.join(base_dest_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    cat_train_dir = os.path.join(train_dir, 'cats/')
    os.makedirs(cat_train_dir, exist_ok=True)
    dog_train_dir = os.path.join(train_dir, 'dogs/')
    os.makedirs(dog_train_dir, exist_ok=True)

    validation_dir = os.path.join(base_dest_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    cat_validation_dir = os.path.join(validation_dir, 'cats/')
    os.makedirs(cat_validation_dir, exist_ok=True)
    dog_validation_dir = os.path.join(validation_dir, 'dogs/')
    os.makedirs(dog_validation_dir, exist_ok=True)

    split_data(cat_source_dir, cat_train_dir, cat_validation_dir, 0.9)
    split_data(dog_source_dir, dog_train_dir, dog_validation_dir, 0.9)


def download_history():
    with open('history_augmented.pkl', 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    if DOWNLOAD:
        if DOWNLOAD_FULL:
            download_and_unzip(FULL_DATA_PATH, 'cats_vs_dogs')
        else:
            download_and_unzip(SMALL_DATA_PATH, 'cats_and_dogs_filtered')

    make_directory_structure()

    if PLOT_IMAGES:
        plot_images('./cats_vs_dogs/train/cats', './cats_vs_dogs/train/dogs', 16)

    train_gen, validation_gen = get_generators('./cats_vs_dogs/train', './cats_vs_dogs/validation')
    model = generate_model()
    history = train(model, train_gen, validation_gen, 1)
    plot_history(history)
    download_history()

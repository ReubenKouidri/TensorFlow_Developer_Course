import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from absl import logging
logging.set_verbosity(logging.ERROR)


def parse_data_from_file(csvfile):
    times = []
    temps = []
    with open(csvfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, row in enumerate(reader):
            times.append(i)
            temps.append(float(row[1]))
    return times, temps


def normalize_data(series):
    min_val = np.min(series)
    max_val = np.max(series)
    normalized_series = (series - min_val) / (max_val - min_val)
    return normalized_series


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


@dataclass
class Globals:
    TEMPERATURES_CSV = './daily-min-temperatures.csv'
    times, temperatures = parse_data_from_file(TEMPERATURES_CSV)
    TIME = np.array(times)
    SERIES = np.array(temperatures)
    SPLIT_TIME = 2500
    WINDOW_SIZE = 64
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    EPOCHS = 50


def train_val_split(time, series, time_step=Globals.SPLIT_TIME):
    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]
    return time_train, series_train, time_valid, series_valid


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


def create_uncompiled_model():
    filters = 64
    kernel_size = 10
    activation = "relu"
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=filters,
                               kernel_size=kernel_size,
                               activation=activation,
                               padding='causal',
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 20)
    ])
    return model


def optimize_lr(optimizers, loss, metrics):
    min_lr = 1e-6
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    max_loss = 0
    for o in optimizers:
        tf.keras.backend.clear_session()
        model = create_uncompiled_model()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: min_lr * 10 ** (epoch / 10))
        model.compile(loss=loss, optimizer=o, metrics=metrics)
        history = model.fit(train_set, epochs=Globals.EPOCHS, callbacks=[lr_schedule])
        lrs = min_lr * (10 ** (np.arange(Globals.EPOCHS) / 20))
        plt.semilogx(lrs, history.history["loss"], label=o.__class__.__name__)
        plt.tick_params('both', length=10, width=1, which='both')
        max_loss = np.max(history.history["loss"] + [max_loss])
        del model

    plt.legend(loc='upper right')
    plt.axis((min_lr, (min_lr * 10 ** (Globals.EPOCHS / 20)), 0, max_loss))
    plt.show()


def compute_metrics(true_series, forecast):
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
    return mse, mae


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


if __name__ == '__main__':
    # time_train, series_train, time_valid, series_valid = train_val_split(Globals.TIME, Globals.SERIES)
    min_val = np.min(Globals.SERIES)
    max_val = np.max(Globals.SERIES)
    normalized_series = (Globals.SERIES - min_val) / (max_val - min_val)
    time_train, normalized_series_train, time_valid, normalized_series_valid = train_val_split(Globals.TIME, normalized_series)

    train_set = windowed_dataset(normalized_series_train,
                                 window_size=Globals.WINDOW_SIZE,
                                 batch_size=Globals.BATCH_SIZE,
                                 shuffle_buffer=Globals.SHUFFLE_BUFFER_SIZE)

    # model = create_uncompiled_model()
    metrics = ["mse", "mae"]
    loss = tf.keras.losses.Huber()
    # lr = 1e-5
    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda e: lr * np.exp(-0.01 * e) if e > 20 else lr)
    optimizer1 = tf.keras.optimizers.legacy.Adam()
    optimizer2 = tf.keras.optimizers.legacy.Nadam()
    optimizer3 = tf.keras.optimizers.legacy.SGD(momentum=0.9)
    optimizers = [optimizer1, optimizer2, optimizer3]

    optimize_lr(optimizers, loss, metrics)
    # model.compile(loss=loss,
    #               optimizer=optimizer,
    #               metrics=metrics)
    #
    # model.fit(train_set, epochs=Globals.EPOCHS)
    #
    # rnn_forecast = model_forecast(model, normalize_data(Globals.SERIES), Globals.WINDOW_SIZE).squeeze()
    # # Slice the forecast to get only the predictions for the validation set
    # rnn_forecast = (rnn_forecast[Globals.SPLIT_TIME - Globals.WINDOW_SIZE: -1] * (max_val - min_val)) + min_val
    # # Plot the forecast
    # plt.figure(figsize=(10, 6))
    # plot_series(time_valid, (normalized_series_valid * (max_val - min_val) + min_val))
    # plot_series(time_valid, rnn_forecast)
    # plt.show()

































import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from absl import logging

logging.set_verbosity(logging.ERROR)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """An arbitrary pattern"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def generate_time_series():
    # The time dimension or the x-coordinate of the time series
    time = np.arange(4 * 365 + 1, dtype="float32")

    # Initial series is just a straight line with a y-intercept
    y_intercept = 10
    slope = 0.005
    series = trend(time, slope) + y_intercept

    # Adding seasonality
    amplitude = 50
    series += seasonality(time, period=365, amplitude=amplitude)

    # Adding some noise
    noise_level = 3
    series += noise(time, noise_level, seed=51)

    return time, series


# Save all "global" variables within the G class (G stands for global)
@dataclass
class Globals:
    TIME, SERIES = generate_time_series()
    SPLIT_TIME = 1100
    WINDOW_SIZE = 20
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    EPOCHS = 50


def train_val_split(time, series, time_step=Globals.SPLIT_TIME):
    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_train, series_train, time_valid, series_valid


def windowed_dataset(series, window_size=Globals.WINDOW_SIZE, batch_size=Globals.BATCH_SIZE,
                     shuffle_buffer=Globals.SHUFFLE_BUFFER_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def create_uncompiled_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                               input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100.0)
    ])
    return model


def create_model():
    model = create_uncompiled_model()
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  metrics=["mae"])
    return model


def find_optimal_loss_long(lr_min=1e-6, lr_max=1e-3):
    """ This function gives a better estimate of the optimal loss but takes much longer to run """
    dataset = windowed_dataset(Globals.SERIES)
    step_size = (lr_max - lr_min) / Globals.EPOCHS
    lrs = np.arange(lr_min, lr_max, step_size)
    min_losses = []
    model = create_uncompiled_model()
    for lr in lrs:
        tf.keras.backend.clear_session()
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.9)
        model.compile(loss='mse', optimizer=optimizer)
        history = model.fit(dataset, epochs=Globals.EPOCHS)
        min_losses.append(min(history.history['loss']))

    return lrs, min_losses


def plot_lr_against_loss(dataset):
    """ This version gives a decent estimate of the optimal loss within 50 epochs and is much faster than the above """
    model = create_uncompiled_model()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch / 20))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis((1e-6, 1, 0, 30))
    plt.show()


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


if __name__ == '__main__':
    time_train, series_train, time_valid, series_valid = train_val_split(Globals.TIME, Globals.SERIES)
    dataset = windowed_dataset(series_train)
    model = create_model()
    model.fit(dataset, epochs=Globals.EPOCHS)
    rnn_forecast = model_forecast(model, Globals.SERIES, Globals.WINDOW_SIZE).squeeze()
    # Slice the forecast to get only the predictions for the validation set
    rnn_forecast = rnn_forecast[Globals.SPLIT_TIME - Globals.WINDOW_SIZE: -1]

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, series_valid)
    plot_series(time_valid, rnn_forecast)
    plt.show()

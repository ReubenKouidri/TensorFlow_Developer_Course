import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass


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


def windowed_dataset(series,
                     window_size=Globals.WINDOW_SIZE,
                     batch_size=Globals.BATCH_SIZE,
                     shuffle_buffer=Globals.SHUFFLE_BUFFER_SIZE
                     ):
    dataset = tf.data.Dataset.from_tensor_slices(series)  # Create dataset from the series
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)  # Slice into the appropriate windows
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))  # Flatten the dataset
    dataset = dataset.shuffle(shuffle_buffer)  # Shuffle it
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))  # Split it into the features and labels
    dataset = dataset.batch(batch_size)  # Batch it
    return dataset


def create_model(window_size=Globals.WINDOW_SIZE):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=4e-5, momentum=0.9)
    model.compile(loss='mse',
                  optimizer=optimizer)
    return model


# def plot_lr_against_loss(dataset):
#     model = create_model()
#     lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
#     history = model.fit(dataset, epochs=Globals.EPOCHS, callbacks=[lr_schedule])
#
#     lrs = 1e-8 * (10 ** (np.arange(Globals.EPOCHS) / 20))
#     plt.figure(figsize=(10, 6))
#     plt.grid(True)
#     plt.semilogx(lrs, history.history["loss"])
#     plt.tick_params('both', length=10, width=1, which='both')
#     plt.axis((1e-8, 1e-3, 0, 300))
#     plt.show()


def find_optimal_loss(lr_min=1e-6, lr_max=1e-3):
    dataset = windowed_dataset(Globals.SERIES)
    step_size = (lr_max - lr_min) / Globals.EPOCHS
    lrs = np.arange(lr_min, lr_max, step_size)
    min_losses = []
    for lr in lrs:
        model = create_model()
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.9)
        model.compile(loss='mse', optimizer=optimizer)
        history = model.fit(dataset, epochs=Globals.EPOCHS)
        min_losses.append(min(history.history['loss']))

    return lrs, min_losses


def generate_forecast(model, series=Globals.SERIES, split_time=Globals.SPLIT_TIME, window_size=Globals.WINDOW_SIZE):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]
    return results


if __name__ == "__main__":
    time_train, series_train, time_valid, series_valid = train_val_split(Globals.TIME, Globals.SERIES)
    dataset = windowed_dataset(series_train)

    model = create_model()
    history = model.fit(dataset, epochs=Globals.EPOCHS)
    # loss = history.history['loss']
    # plt.plot(np.arange(Globals.EPOCHS), loss, 'b', label='Training Loss')
    # plt.show()

    dnn_forecast = generate_forecast(model)
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, series_valid)
    plot_series(time_valid, dnn_forecast)
    plt.show()

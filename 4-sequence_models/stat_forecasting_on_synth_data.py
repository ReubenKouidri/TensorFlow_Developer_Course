from __future__ import annotations
from typing import Optional
import collections.abc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, fmt: str = "-", start: int = 0, end: int = -1) -> None:
    """
    Plot a series of values over time.

    Parameters:
        time (list | tuple): A list of time values.
        series (list | tuple): The series of values to be plotted. If a tuple is provided, each element of the tuple will be plotted separately.
        fmt (str, optional): The format of the plot lines. Defaults to "-".
        start (int, optional): The starting index of the series to be plotted. Defaults to 0.
        end (int, optional): The ending index of the series to be plotted. If None, the entire series will be plotted. Defaults to None.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    if isinstance(series[0], collections.abc.Iterable):  # series of series
        for series_num in series:
            plt.plot(time[start:end], series_num[start:end], fmt)
    else:
        plt.plot(time[start:end], series[start:end], fmt)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def trend(time: np.ndarray, slope=0) -> np.ndarray:
    """
    Generates synthetic data that follows a straight line given a slope value.

    Args:
      time (array of int) - contains the time steps
      slope (float) - determines the direction and steepness of the line

    Returns:
      series (array of float) - measurements that follow a straight line
    """
    series = slope * time
    return series


def seasonal_pattern(season_time: np.ndarray) -> np.ndarray:
    """
    Just an arbitrary pattern, you can change it if you wish

    Args:
      season_time (array of float) - contains the measurements per time step

    Returns:
      data_pattern (array of float) -  contains revised measurement values according
                                  to the defined pattern
    """
    # Generate the values using an arbitrary pattern
    data_pattern = np.where(season_time < 0.4,
                            np.cos(season_time * 2 * np.pi),
                            1 / np.exp(3 * season_time))
    return data_pattern


def seasonality(time, period, amplitude=1, phase=0):
    """
    Repeats the same pattern at each period

    Args:
      time (array of int) - contains the time steps
      period (int) - number of time steps before the pattern repeats
      amplitude (int) - peak measured value in a period
      phase (int) - number of time steps to shift the measured values

    Returns:
      data_pattern (array of float) - seasonal data scaled by the defined amplitude
    """
    # Define the measured values per period
    season_time = ((time + phase) % period) / period
    # Generates the seasonal data scaled by the defined amplitude
    data_pattern = amplitude * seasonal_pattern(season_time)
    return data_pattern


def noise(time, noise_level=1, seed=None):
    """
    Generates a normally distributed noisy signal

    Args:
      time (array of int) - contains the time steps
      noise_level (float) - scaling factor for the generated signal
      seed (int) - number generator seed for repeatability

    Returns:
      noise (array of float) - the noisy signal
    """
    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level
    return noise


def generate_synth_data(num_periods: int,
                        len_periods: int,
                        baseline: Optional[int] = 40,
                        amplitude: Optional[int] = 40,
                        slope: Optional[float] = 0.05,
                        noise_level: Optional[float] = 5.0
                        ) -> tuple:
    time = np.arange(num_periods * len_periods + 1, dtype="float32")
    series = baseline + trend(time, slope) + seasonality(time, period=len_periods, amplitude=amplitude)
    series += noise(time, noise_level, seed=42)
    return series, time


def split_data(series, split_time=1000):
    x_train = series[:split_time]
    time_train = time[:split_time]
    x_valid = series[split_time:]
    time_valid = time[split_time:]
    return x_train, time_train, x_valid, time_valid


def moving_average_forecast(series, window_size) -> np.ndarray:
    """Generates a moving average forecast

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to compute the average for

    Returns:
      forecast (array of float) - the moving average forecast
    """
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time: time + window_size].mean())
    return np.array(forecast)


if __name__ == "__main__":
    split_time = 1000
    series, time = generate_synth_data(num_periods=4, len_periods=365)
    x_train, train_time, x_valid, time_valid = split_data(series, split_time=split_time)
    # plot_series(train_time, x_train, fmt="-")
    # plot_series(time_valid, x_valid, fmt="-")

    # naive_forecast = series[split_time - 1: -1]
    # plot_series(time_valid, (x_valid, naive_forecast))
    # plot_series(time_valid, (x_valid, naive_forecast), start=0, end=150)

    # Generate the moving average forecast
    moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

    # Plot the results
    # plot_series(time_valid, (x_valid, moving_avg))
    # print(tf.keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
    # print(tf.keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())

    diff_series = (series[365:] - series[:-365])
    # Truncate the first 365 time steps
    diff_time = time[365:]
    plot_series(diff_time, diff_series)
    diff_moving_avg = moving_average_forecast(diff_series, 30)
    # Slice the prediction points that corresponds to the validation set time steps
    diff_moving_avg = diff_moving_avg[split_time - 365 - 30:]
    # Slice the ground truth points that corresponds to the validation set time steps
    diff_series = diff_series[split_time - 365:]
    plot_series(time_valid, (diff_series, diff_moving_avg))
    diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
    plot_series(time_valid, (x_valid, diff_moving_avg_plus_past))
    print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
    print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
    # Smooth the original series before adding the time differenced moving average
    diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-359], 11) + diff_moving_avg

    # Plot the results
    plot_series(time_valid, (x_valid, diff_moving_avg_plus_smooth_past))
    # Compute the metrics
    print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
    print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())

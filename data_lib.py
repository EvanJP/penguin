import dataclasses
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


@dataclasses.dataclass
class AlphaVantageConfig():
    """Config for AlphaVantage access"""
    api_key: str
    symbol: str
    outputsize: str
    keys: dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DataConfig():
    window_size: int
    train_split: float
    batch_size: int
    gamma: float


@dataclasses.dataclass
class MatplotlibConfig():
    xticks_interval: int = 90  # show a date every 90 days
    color_actual: str = "#001f3f"
    color_train: str = "#3D9970"
    color_val: str = "#0074D9"
    color_pred_train: str = "#3D9970"
    color_pred_val: str = "#0074D9"
    color_pred_test: str = "#FF4136"


def prepare_data(data_df, config: DataConfig):
    data = data_df.loc[:, 'adjusted close'].to_numpy()
    split_index = int(len(data) * config.train_split)
    train_data = data[:split_index].reshape(-1, 1)
    test_data = data[split_index:].reshape(-1, 1)
    scaler = MinMaxScaler()

    for di in range(0, len(train_data), config.window_size):
        window = di + config.window_size
        train_data[di: window, :] = scaler.fit_transform(
            train_data[di: window, :])

    train_data = train_data.reshape(-1)
    test_data = scaler.transform(test_data).reshape(-1)

    # Smooth to Exponential Moving Average.
    EMA = 0.0
    for point in range(len(train_data)):
        EMA = config.gamma * train_data[point] + (1-config.gamma) * EMA
        train_data[point] = EMA

    return train_data, test_data


def download_data(config: AlphaVantageConfig):
    file_path = os.path.join(
        os.getcwd(),
        f'stock_market_data_{config.symbol}_ticker.csv')
    print(file_path)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    columns = ['date'] + [feature.split(' ', 1)[1] for feature in config.keys]
    df = pd.DataFrame(columns=columns)
    ts = TimeSeries(key=config.api_key)
    data, _ = ts.get_daily_adjusted(
        config.symbol, outputsize=config.outputsize)
    df.loc[:, 'date'] = list(
        reversed(
            [datetime.strptime(date, '%Y-%m-%d')
             for date in data.keys()]))
    for feature in config.keys:
        df.loc[:, feature.split(' ', 1)[1]] = np.array(list(reversed([
            float(data[date][feature]) for date in data.keys()])))
    df.to_csv(file_path)
    return df


def plot_data(df):
    plt.figure(figsize=(20, 10))
    plt.plot(range(df.shape[0]), (df['low'] + df['high'])/2.0)
    plt.xticks(range(0, df.shape[0], 500), df['date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()


class DataGeneratorSeq:
    """Data generator for time series stock prices.

    Given `prices`, batch them into `num_unroll` sets of size `batch_size`.

    E.g. 
    `num_unroll = 1`, `batch_size = 4`, 
    `input = [x_0,x_10,x_20,x_30]`, 
    `output = [x_1,x_11,x_21,x_31]`
    """

    def __init__(self, prices, batch_size, num_unroll, window_size=5):
        """Data generator for time series stock prices.

        Args:
            prices: The data to segment.
            batch_size: How big each batch should be.
            num_unroll: How many batches to make.
            window_size: How far into the future each output should be sampled 
            from.
        """
        self.prices = prices
        # make sure cursor doesn't go OOB.
        self.prices_length = len(prices) - num_unroll
        self.batch_size = batch_size
        self.num_unroll = num_unroll
        self.segments = self.prices_length // batch_size
        # Pointers to start of each data batch.
        self.cursor = [offset * self.segments
                       for offset in range(batch_size)]
        self.window_size = window_size
        self.rng = np.random.default_rng()

    def next_batch(self):
        """Generate batch of data.

        To make more robust, make `x_t` output not `x_{t+1}`, 
        but rather a random sample from `x_{t+1}, x_{t+2}, ..., x_{t+N}`
        """
        batch_data = np.zeros(self.batch_size, dtype=np.float32)
        batch_labels = np.zeros(self.batch_size, dtype=np.float32)
        for b in range(self.batch_size):
            if self.cursor[b] + 1 >= self.prices_length:
                self.cursor[b] = self.rng.integers(0, (b+1) * self.segments)
            batch_data[b] = self.prices[self.cursor[b]]
            batch_labels[b] = self.prices[self.cursor[b]
                                          + self.rng.integers(0, self.window_size)]
            self.cursor[b] = (self.cursor[b] + 1) % self.prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data = []
        unroll_labels = []
        for _ in range(self.num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self.batch_size):
            self.cursor[b] = self.rng(
                0, min((b+1) * self.segments, self.prices_length-1))

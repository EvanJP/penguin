import dataclasses
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import data_lib
import tensorflow as tf
import model_lib


config = {
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,  # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}


def main():
    key = "ZXKQBF8XD8T4KLQG"
    plot_data_trigger = False
    alpha_vantage_config = data_lib.AlphaVantageConfig(
        api_key=key, symbol="IBM", outputsize="full",
        keys=["3. low", "2. high", "5. adjusted close", "1. open"])
    print('hello')
    data_df = data_lib.download_data(alpha_vantage_config)
    if plot_data_trigger:
        data_lib.plot_data(data_df)

    data_config = data_lib.DataConfig(
        window_size=1_000, train_split=0.8, batch_size=64, gamma=0.1)
    train_data, test_data = data_lib.prepare_data(data_df, data_config)
    data_generator = data_lib.DataGeneratorSeq(train_data, 5, 5)
    u_data, u_labels = data_generator.unroll_batches()
    for i, (dat, lbl) in enumerate(zip(u_data, u_labels)):
        print('\n\nUnrolled index %d' % i)
        print('\t Inputs: ', dat)
        print('\n\t Output: ', lbl)
    tf.reset_default_graph()
    # model = model_lib.create_time_2_vector_transformer
    # train, val = data_lib.get_data_loaders(
    #     alpha_vantage_config, matplotlib_config, data_config)


if __name__ == "__main__":
    main()

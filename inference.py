import os
import time
import ncps.wirings
import pandas as pd
import keras
from keras import layers
from keras import losses, optimizers
from ncps.keras import CfC
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import datetime

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 500)


def create_X(funcdata, feats_num, window=3):
    X = []
    for i in range(len(funcdata)):
        X.append(funcdata[i:i + window])
    return np.array(X).reshape((-1, window, feats_num))


def load_scalers(filenames, stock_name):
    func_scalers = dict()
    for scaler_file in filenames:
        func_scalers[scaler_file.split(".")[0]] = joblib.load(f"scalers/{stock_name}/{scaler_file}")
    return func_scalers


def restore_model(stock_name: str, input_shape: tuple, conv_neurons: int,
                  pool_neurons: int, pool_output=2, sparsity=0.5):
    inp = layers.Input(shape=input_shape)
    conv = layers.Conv1D(conv_neurons, 1, activation=layers.LeakyReLU(0.4))(inp)

    wiring = ncps.wirings.AutoNCP(pool_neurons, pool_output, sparsity, seed=42)
    pool = CfC(units=wiring, mixed_memory=True, return_sequences=False, stateful=False)(conv)

    out = layers.Dense(1, activation=None)(pool)

    func_model = keras.Model(inputs=inp, outputs=out)
    func_model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.MeanSquaredError())

    func_model.load_weights(f'weights/{stock_name}.weights.h5')
    return func_model


# Создание входных данных - часть проекта другого участника
def create_df():
    return pd.DataFrame()


def routine():
    in_window = 3
    units = 96
    neur_conv = 49

    stocks = []

    for stock in stocks:
        scaler_files = os.listdir(f"{os.curdir}/scalers/{stock}/")
        num_of_feats = len(scaler_files)

        scalers = load_scalers(scaler_files, stock)

        data = create_df()

        for scaler_name in scalers:
            data[scaler_name] = scalers[scaler_name].transform(data[scaler_name].values.reshape((-1, 1)))

        # Порядок данных в дф

        X = create_X(data, num_of_feats, in_window)

        model = restore_model(stock, (in_window, num_of_feats), neur_conv, units)

        output = model.predict(X)

        output = scalers["RSI"].inverse_transform(output)
        print(f"Next hour's RSI for {stock}: {output}")


def main():
    while True:
        now = datetime.datetime.now()
        if now.minute == 1:
            routine()
            time.sleep(60)
        else:
            time.sleep(10)


if __name__ == "__main__":
    main()

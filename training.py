import os
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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 500)


def load_df(name: str):
    func_data = pd.read_csv("sets/hourly/" + name)
    func_data = func_data.set_index("Date").drop(columns=["ticker", "end"])
    func_data = func_data[50:]
    func_feats = func_data.columns.tolist()
    num = len(func_feats)
    return func_data, func_feats, num


def create_XY(funcdata: pd.DataFrame, feats_num: int, window=3):
    X = []
    Y = []
    for i in range(len(funcdata) - window):
        X.append(funcdata[i:i + window])
        Y.append(funcdata['RSI'].iloc[i + window])
    return np.array(X).reshape((-1, window, feats_num)), np.array(Y).reshape((-1, 1))


def split_df(df: pd.DataFrame, train_border=0.65, val_border=0.2):
    b1 = int(len(df) * train_border)
    b2 = int(len(df) * (train_border + val_border))
    return df[:b1], df[b2:], df[b1:b2]


def scale(func_train_data: pd.DataFrame, func_val_data: pd.DataFrame,
          func_test_data: pd.DataFrame, func_feats: list):
    func_scalers = dict()
    for feat in func_feats:
        scaler = StandardScaler()
        func_train_data[feat] = scaler.fit_transform(func_train_data[feat].values.reshape((-1, 1)))
        func_val_data[feat] = scaler.transform(func_val_data[feat].values.reshape((-1, 1)))
        func_test_data[feat] = scaler.transform(func_test_data[feat].values.reshape((-1, 1)))
        func_scalers[feat] = scaler
    return func_train_data, func_val_data, func_test_data, func_scalers


def save_scalers(stock_name: str, func_scalers: dict):
    for scaler_name in func_scalers:
        joblib.dump(func_scalers[scaler_name], f"scalers/{stock_name}/{scaler_name}.save")


def create_model(input_shape: tuple, conv_neurons: int, pool_neurons: int,
                 pool_output=2, sparsity=0.5, rate=0.0001):
    inp = layers.Input(shape=input_shape)
    conv = layers.Conv1D(conv_neurons, 1, activation=layers.LeakyReLU(0.4))(inp)

    wiring = ncps.wirings.AutoNCP(pool_neurons, pool_output, sparsity, seed=42)
    pool = CfC(units=wiring, mixed_memory=True, return_sequences=False, stateful=False)(conv)

    out = layers.Dense(1, activation=None)(pool)

    func_model = keras.Model(inputs=inp, outputs=out)
    func_model.compile(optimizer=optimizers.Adam(learning_rate=rate),
                       loss=losses.MeanSquaredError())
    return func_model


def main():
    in_window = 3
    train_p = 0.65
    val_p = 0.2

    units = 96
    neur_conv = 49
    lr = 0.0001
    b_size = 32

    sets = os.listdir(f"{os.curdir}/sets/hourly/")
    for en in sets:
        stock = en.split("_")[0]
        print(stock)

        data, feats, num_of_feats = load_df(en)

        train_data, val_data, test_data = split_df(data, train_p, val_p)

        train_data, val_data, test_data, scalers = scale(train_data, val_data, test_data, feats)

        save_scalers(stock, scalers)

        Xtrain, Ytrain = create_XY(train_data, num_of_feats, in_window)
        Xval, Yval = create_XY(val_data, num_of_feats, in_window)
        Xtest, Ytest = create_XY(test_data, num_of_feats, in_window)

        model = create_model((in_window, num_of_feats), neur_conv, units, rate=lr)
        print(model.summary())

        model.fit(x=Xtrain, y=Ytrain, validation_data=(Xval, Yval), batch_size=b_size, epochs=300, shuffle=False,
                  callbacks=[
                      keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
                      keras.callbacks.EarlyStopping(patience=15)
                  ]
                  )

        print(model.evaluate(Xtest, Ytest))

        test_out = model.predict(Xtest)
        test_out = scalers["RSI"].inverse_transform(test_out.reshape((-1, 1))).reshape((-1,))
        Ytest = scalers["RSI"].inverse_transform(Ytest.reshape((-1, 1))).reshape((-1,))

        t = np.arange(1, len(Ytest) + 1)
        plt.plot(t, test_out, "b", label="Output")
        plt.plot(t, Ytest, "g", label="Actual")
        plt.title(f"{stock} RSI")
        plt.legend()
        plt.show()

        model.save_weights(f"weights/{stock}.weights.h5")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

np.random.seed(0)

def generate_temperatures():
    # 月平均気温
    temperatures = [[7.6, 6.0, 9.4, 14.5, 19.8, 22.5, 27.7, 28.3, 25.6, 18.8, 13.3, 8.8], #2000年
                    [4.9, 6.6, 9.8, 15.7, 19.5, 23.1, 28.5, 26.4, 23.2, 18.7, 13.1, 8.4], #2001年
                    [7.4, 7.9, 12.2, 16.1, 18.4, 21.6, 28.0, 28.0, 23.1, 19.0, 11.6, 7.2],#2002年
                    [5.5, 6.4, 8.7, 15.1, 18.8, 23.2, 22.8, 26.0, 24.2, 17.8, 14.4, 9.2], #2003年
                    [6.3, 8.5, 9.8, 16.4, 19.6, 23.7, 28.5, 27.2, 25.1, 17.5, 15.6, 9.9], #2004年
                    [6.1, 6.2, 9.0, 15.1, 17.7, 23.2, 25.6, 28.1, 24.7, 19.2, 13.3, 6.4],
                    [5.1, 6.7, 9.8, 13.6, 19.0, 22.5, 25.6, 27.5, 23.5, 19.5, 14.4, 9.5],
                    [7.6, 8.6, 10.8, 13.7, 19.8, 23.2, 24.4, 29.0, 25.2, 19.0, 13.3, 9.0],
                    [5.9, 5.5, 10.7, 14.7, 18.5, 21.3, 27.0, 26.8, 24.4, 19.4, 13.1, 9.8],
                    [6.8, 7.8, 10.0, 15.7, 20.1, 22.5, 26.3, 26.6, 23.0, 19.0, 13.5, 9.0],
                    [7.0, 6.5, 9.1, 12.4, 19.0, 23.6, 28.0, 29.6, 25.1, 18.9, 13.5, 9.9],
                    [5.1, 7.0, 8.1, 14.5, 18.5, 22.8, 27.3, 27.5, 25.1, 19.5, 14.9, 7.5],
                    [4.8, 5.4, 8.8, 14.5, 19.6, 21.4, 26.4, 29.1, 26.2, 19.4, 12.7, 7.3],
                    [5.5, 6.2, 12.1, 15.2, 19.8, 22.9, 27.3, 29.2, 25.2, 19.8, 13.5, 8.3],
                    [6.3, 5.9, 10.4, 15.0, 20.3, 23.4, 26.8, 27.7, 23.2, 19.1, 14.2, 6.7],
                    [5.8, 5.7, 10.3, 14.5, 21.1, 22.1, 26.2, 26.7, 22.6, 18.4, 13.9, 9.3],
                    [6.1, 7.2, 10.1, 15.4, 20.2, 22.4, 25.4, 27.1, 24.4, 18.7, 11.4, 8.9]]#2016年
    temperatures = np.array(temperatures)
    temperatures = np.reshape(temperatures, (temperatures.size))
    return temperatures


def generate_data(temperatures, length_per_unit, dimension):
    sequences = []
    target = []
    for i in range(0, temperatures.size - length_per_unit):
        sequences.append(temperatures[i:i + length_per_unit])
        target.append(temperatures[i + length_per_unit])

    X = np.array(sequences).reshape(len(sequences), length_per_unit, dimension)
    Y = np.array(target).reshape(len(sequences), dimension)

    N_train = int(len(sequences) * 0.9)
    X_train = X[:N_train]
    X_validation = X[N_train:]
    Y_train = Y[:N_train]
    Y_validation = Y[N_train:]

    return (X_train, X_validation, Y_train, Y_validation)


def build_model(input_shape, hidden_layer_count):
    model = Sequential()
    model.add(SimpleRNN(hidden_layer_count, input_shape=input_shape))
    model.add(Dense(input_shape[1]))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


# 一つの時系列データの長さ
LENGTH_PER_UNIT = 24
# 一次元データを扱う
DIMENSION = 1
# 年別月平均気温の生成
temperatures = generate_temperatures()
# トレーニング、バリデーション用データの生成
X_train, X_validation, Y_train, Y_validation = generate_data(temperatures, LENGTH_PER_UNIT, DIMENSION)

# SimpleRNN隠れ層の数
HIDDEN_LAYER_COUNT = 25
# 入力の形状
input_shape=(LENGTH_PER_UNIT, DIMENSION)
# モデルの生成
model = build_model(input_shape, HIDDEN_LAYER_COUNT)

# モデルのトレーニング
epochs = 500
batch_size = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])

# 予測を行う
part_of_sequence = np.array([temperatures[i] for i in range(LENGTH_PER_UNIT)])
predicted = [None for i in range(LENGTH_PER_UNIT)]
Z = X_train[:1, :, :]
for i in range(temperatures.size - LENGTH_PER_UNIT + 1):
    y_ = model.predict(Z)
    # 予測結果を入力として利用するため、第0項を削除し予測結果をひっつける
    Z = np.concatenate(
            (Z.reshape(LENGTH_PER_UNIT, DIMENSION)[1:], y_),
            axis=0).reshape(1, LENGTH_PER_UNIT, DIMENSION)
    predicted.append(y_.reshape(-1))
predicted = np.array(predicted)

# 予測結果の描画
plt.rc('font', family='serif')
plt.figure()
plt.ylim([0.0, 35.0])
plt.plot(temperatures, linestyle='dotted', color='#aaaaaa')
plt.plot(part_of_sequence, linestyle='dashed', color='blue')
plt.plot(predicted, color='red')
plt.show()

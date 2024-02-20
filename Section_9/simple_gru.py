import numpy as np
import matplotlib.pyplot as plt

# !訓練用データの作成
# ?サイン関数に乱数でノイズを加えたデータを作成し、過去の時系列データから未来の値を予測できるようにします。
# ?今回も、最後の時刻の出力のみ利用します。
x_data = np.linspace(-2*np.pi, 2*np.pi)  # -2πから2πまで
sin_data = np.sin(x_data) + 0.1*np.random.randn(len(x_data))  # sin関数に乱数でノイズを加える

plt.plot(x_data, sin_data)
plt.show()

n_rnn = 10  # 時系列の数
n_sample = len(x_data)-n_rnn  # サンプル数
x = np.zeros((n_sample, n_rnn))  # 入力
t = np.zeros((n_sample,))  # 正解、最後の時刻のみ

for i in range(0, n_sample):
    x[i] = sin_data[i:i+n_rnn]
    t[i] = sin_data[i+n_rnn]  # 入力の時系列より一つ後の値

x = x.reshape(n_sample, n_rnn, 1)  # （サンプル数、時系列の数、入力層のニューロン数）
print(x.shape)
t = t.reshape(n_sample, 1)  # （サンプル数、入力層のニューロン数）
print(t.shape)

# !LSTMとGRUの比較
# ?Kerasを使ってLSTM、およびGRUを構築します。
# ?Kerasにおいて、GRU層はSimpleRNN層やLSTM層と同じ方法で追加することができます。
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU

batch_size = 8
n_in = 1
n_mid = 20
n_out = 1

# 比較のためのLSTM
model_lstm = Sequential()
model_lstm.add(LSTM(n_mid, input_shape=(n_rnn, n_in), return_sequences=False))
model_lstm.add(Dense(n_out, activation="linear"))
model_lstm.compile(loss="mean_squared_error", optimizer="sgd")
print(model_lstm.summary())

# GRU
model_gru = Sequential()
model_gru.add(GRU(n_mid, input_shape=(n_rnn, n_in), return_sequences=False))
model_gru.add(Dense(n_out, activation="linear"))
model_gru.compile(loss="mean_squared_error", optimizer="sgd")
print(model_gru.summary())

# TODO LSTMよりも、GRUの方がパラメータが少ないですね。

# ! 学習
# ?構築したRNNのモデルを使って、学習を行います。
import time

epochs = 300

# LSTM
start_time = time.time()
history_lstm = model_lstm.fit(x, t, epochs=epochs, batch_size=batch_size, verbose=0)
print("学習時間 --LSTM--:", time.time() - start_time)

# GRU
start_time = time.time()
history_gru = model_gru.fit(x, t, epochs=epochs, batch_size=batch_size, verbose=0)
print("学習時間 --GRU--:", time.time() - start_time)

# TODO エポック数が同じ場合、パラメータ数が少ないためGRUの方が学習に時間がかかりません。

# !学習の推移
# ?誤差の推移を確認します。
loss_lstm = history_lstm.history['loss']
loss_gru = history_gru.history['loss']

plt.plot(np.arange(len(loss_lstm)), loss_lstm, label='LSTM')
plt.plot(np.arange(len(loss_gru)), loss_gru, label='GRU')
plt.legend()
plt.show()

# TODO 最初は早く収束するのですが、最終的な収束にはGRUの方がエポック数が必要なようです。



# !学習済みモデルの使用
# ?それぞれの学習済みモデルを使って、サイン関数の次の値を予測します。
predicted_lstm = x[0].reshape(-1)
predicted_gru = x[0].reshape(-1)

for i in range(0, n_sample):
    y_lstm = model_lstm.predict(predicted_lstm[-n_rnn:].reshape(1, n_rnn, 1))
    predicted_lstm = np.append(predicted_lstm, y_lstm[0][0])
    y_gru = model_gru.predict(predicted_gru[-n_rnn:].reshape(1, n_rnn, 1))
    predicted_gru = np.append(predicted_gru, y_gru[0][0])

plt.plot(np.arange(len(sin_data)), sin_data, label='Training data')
plt.plot(np.arange(len(predicted_lstm)), predicted_lstm, label="Predicted_LSTM")
plt.plot(np.arange(len(predicted_gru)), predicted_gru, label="GRU")
plt.legend()
plt.show()

# TODO GRUを使ったモデルも、LSTMと同様にサインカーブを学習できていることが分かります。
# TODO このグラフでは、LSTMの方がよく収束していますね。
# TODO このように、GRUはパラメータ数が少なく1エポックに必要な時間は短いのですが、LSTMよりも収束にエポック数が必要な場合があります。
# TODO 状況に応じて、LSTMとGRUを使い分ける必要があります。
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


# ! 訓練用データの作成
# ? RNNに用いる訓練用のデータを作成します
# ? サイン関数に乱数でノイズを加えたデータを作成し、過去の時系列データから未来の値を予測できるようにします
# * np.linspace関数は、指定された範囲内で等間隔の数値を生成します。この場合、-2*np.piから2*np.piまでの範囲です。つまり、マイナス2倍の円周率からプラス2倍の円周率までの数値を等間隔で生成しています。これらの数値は、サイン波のグラフを描くための「時間」軸や「角度」軸として使われます
x_data = np.linspace(-2*np.pi, 2*np.pi) # -2πから2πまで
# * np.sin(x_data)は、x_dataの各数値に対してサイン（正弦）関数を適用し、サイン波形を生成します..
# * 0.1*np.random.randn(len(x_data))は、x_dataと同じ長さのランダムな数値の配列を生成し、それを0.1倍しています。これにより、「ノイズ」が作られ、サイン波形に少しの乱れを加えることができます。この「ノイズ」によって、実際のデータが持つような不規則性やばらつきを模倣しています。
sin_data = np.sin(x_data) + 0.1*np.random.randn(len(x_data)) # sin関数に乱数でノイズを加える
# print(x_data)
# print(sin_data)
plt.plot(x_data, sin_data)
# plt.show()


# ? 、時系列データを扱うための準備をしています。時系列データとは、時間の経過とともに観測されたデータのことで、この場合はノイズが加えられたサイン波形のデータを使います。ニューラルネットワーク、特にリカレントニューラルネットワーク（RNN）でこのようなデータを学習させる際には、データを適切に整形する必要があります
# ? 一つのサンプルに含まれる時系列データの点の数です。ここでは、10時点分のデータを一つのサンプルとして扱います
n_rnn = 10 # 時系列の数
# ? 全データの長さから時系列の長さを引いて、生成可能なサンプルの数を計算しています
n_sample = len(x_data) - n_rnn # サンプル数
# ? xはニューラルネットワークに入力されるデータを格納する配列、tはそれに対応する正解データ（ターゲット）を格納する配列です。これらは最初にゼロで初期化されています
x = np.zeros((n_sample, n_rnn)) # 入力
t = np.zeros((n_sample, n_rnn)) # 正解

# ? このループでは、sin_dataから時系列データのサンプルを切り出してxとtに格納しています
# ? xにはある時点から始まるn_rnn個のデータが入り、tにはそれに対応する次の時点から始まるn_rnn個のデータが入ります。これにより、xのデータを使ってtのデータを予測するという学習が可能になります
for i in range(0, n_sample):
    x[i] = sin_data[i:i+n_rnn]
    t[i] = sin_data[i+1:i+n_rnn+1] # 時系列を入力よりも一つ後にずらす

# ? RNNに入力するために、データの形状を変更しています。Kerasでは、RNNの入力データは3次元配列である必要があり、その形状は（サンプル数、時系列の長さ、特徴量の数）です。このケースでは、各時系列データ点が1つの特徴量を持つため、最後の次元は1です
x = x.reshape(n_sample, n_rnn, 1) # KerasにおけるRNNでは、入力を（サンプル数、時系列の数、入力層のニューロン数）にする
print(x.shape)
# ? tも同様に形状を変更しています。
t = t.reshape(n_sample, n_rnn, 1) # 今回は入力と同じ形状
print(t.shape)

# ! RNNの構築
# ? Kerasを使ってRNNを構築
# ? 今回は、Kerasが持つRNNの中で一番シンプルなSimpleRNN層を使う
batch_size = 8
n_in = 1
n_mid = 20
n_out = 1

model = Sequential()
# ? SimpleRNN層をモデルに追加しています。n_midはこの層のニューロン数（20個）を指定
# ? input_shape=(n_rnn, n_in)は、この層の入力の形状を定義
# ? n_rnnは時系列データの長さ
# ? n_inは入力の特徴量の数（1つ）を意味
model.add(SimpleRNN(n_mid, input_shape=(n_rnn, n_in), return_sequences=True))
model.add(Dense(n_out, activation="linear"))
model.compile(loss="mean_squared_error", optimizer="sgd")
print(model.summary())

# ! 学習
history = model.fit(x, t, batch_size=batch_size, epochs=20, validation_split=0.1)

# ! 学習の推移
loss = history.history["loss"]
vloss = history.history["val_loss"]
plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(vloss)), vloss)
plt.show()

# ! 学習モデルの使用
predicted = x[0].reshape(-1) # 最初の入力。reshape(-1)で一次元のベクトルにする。

for i in range(0, n_sample):
    y = model.predict(predicted[-n_rnn:]. reshape(1, n_rnn, 1))  # 直近のデータを使って予測を行う
    predicted = np.arange(predicted, y[0][n_rnn-1][0]) # 出力の最後の結果をpredictedに追加する

plt.plot(np.arange(len(sin_data)), sin_data, label="Training data")
plt.plot(np.arange(len(predicted)), predicted, label="Predicted")
plt.legend()
plt.show()
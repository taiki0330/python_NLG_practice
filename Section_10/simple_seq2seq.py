import numpy as np
import matplotlib.pyplot as plt

"""シンプルなseq2seq¶
最小限のSeq2Seqを構築し、時系列の変換を行います。
今回は、Seq2Seqを使って、sin関数の曲線をcos関数の曲線に”翻訳”します。
Seq2Seqの構築方法について、基礎から学んでいきましょう。"""

"""訓練用データの作成
訓練用のデータを作成します。
今回は、sin関数の値をencoderへの入力、cos関数の値をdecoderへの入力、及び正解とします。
decoderへの入力は、正解から一つ後の時刻にずらします。
これにより、ある時刻におけるdecoderの出力が、次の時刻における入力に近づくように学習を行うことができます。
このような、ある時刻における正解が次の時刻の入力となる手法を教師強制といいます。"""

axis_x = np.linspace(-2*np.pi, 2*np.pi)
# print(axis_x)
sin_data = np.sin(axis_x)
cos_data = np.cos(axis_x)

plt.plot(axis_x, sin_data)
plt.plot(axis_x, cos_data)
# plt.show()

n_rnn = 10 # 時系列の数
n_sample = len(axis_x)-n_rnn #サンプル数
x_encoder = np.zeros((n_sample, n_rnn)) #encoderの入力
x_decoder = np.zeros((n_sample, n_rnn)) #decoderの入力
t_decoder = np.zeros((n_sample, n_rnn)) #decoderの正解

for i in range(0, n_sample):
    x_encoder[i] = sin_data[i:i+n_rnn]
    x_decoder[i, 1:] = cos_data[i:i+n_rnn-1]# 一つ後の時刻にずらす。最初の値は0のまま。
    t_decoder[i] = cos_data[i:i+n_rnn]# 正解は、cos関数の値をそのまま入れる

x_encoder = x_encoder.reshape(n_sample, n_rnn, 1) # （サンプル数、時系列の数、入力層のニューロン数）
x_decoder = x_decoder.reshape(n_sample, n_rnn, 1)
t_decoder = t_decoder.reshape(n_sample, n_rnn, 1)


"""Seq2Seqの構築¶
Kerasを使ってSeq2Seqを構築します。
今回は、Sequentialクラスではなく、Modelクラスを使います。
Modelクラスを使えば、複数の経路の入力を持つニューラルネットワークを構築可能で、状態を渡すことでRNN同士を接続することもできます。
今回は、Seq2SeqのRNN部分にはLSTMを使います。"""
from keras.models import Model
from keras.layers import Dense, LSTM, Input

n_in = 1
n_mid = 20
n_out = n_in

encoder_input = Input(shape=(n_rnn, n_in))
encoder_lstm = LSTM(n_mid, return_state=True)
encoder_output

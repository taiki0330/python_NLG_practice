# import matplotlib
# matplotlib.use('TkAgg')
# NumPyライブラリをインポートしています。これは、数値計算を効率的に行うためのツールを提供しており、npという短縮名で参照できるようになります。
import numpy as np
# Matplotlibのpyplotモジュールをインポートしています。これは、グラフを描くための関数を提供しており、pltという短縮名で参照できるようになります。
import matplotlib.pyplot as plt
# 「Sequential」は頭脳を作るための空の箱
from keras.models import Sequential
# 「Dense」は、ニューロンをぎっしり詰め込んだ層を作るためのもの
from keras.layers import Dense

# ! 以下はkerasの基礎
#  np.linspace関数は、指定された範囲（ここでは −π から π）内の等間隔の数値を生成します。その後、reshape(-1, 1)を使用して、生成されたデータを列ベクトルに変換します。-1はNumPyに配列の新しい形を計算させるためのプレースホルダとして機能します。
x = np.linspace(-np.pi, np.pi).reshape(-1, 1)

# xの各要素に対してサイン関数（正弦関数）を適用しています。この結果、xの各点に対するsinの値が含まれる新しい配列tが作成されます。
t = np.sin(x)
# print(x)

# xのデータを横軸に、tのデータを縦軸に取ってグラフにプロットします。
plt.plot(x, t)

# 作成されたグラフを画面に表示します。
# plt.show()


#! 頭脳の箱
#? バッチサイズは、コンピュータが一度に学習するデータの量です。
batch_size = 8 # バッチサイズ
#? 入力層のニューロン数は、コンピュータに入れる情報の種類の数です。
n_in = 1 # 入力層のニューロン数
#? 中間層のニューロン数は、脳で考えるときに使う「考える力」の部分です。
n_mid = 20 # 中間層のニューロン数
#? 出力層のニューロン数は、学習の結果、コンピュータが出す答えの種類の数です。
n_out = 1 # 出力層のニューロン数


#! 次に、実際にニューラルネットワークを作っています。入力層、中間層、出力層の３層のニューラルネットワークを構築。
# ? 層を積み重ねるモデルを作る
model = Sequential() # ニューロンを繋げる枠組みを作っています
#? たくさんのニューロンを中間層に追加
#? return_sequences=Trueというのは、次にどのように情報を渡すかを決める部分です。
#? シグモイド関数は、ニューロンが「オン」になるか「オフ」になるかを決めるスイッチみたいなもの
model.add(Dense(n_mid, input_shape=(n_in, ), activation="sigmoid")) # 活性化関数にシグモイド関数

#? 最後の層には答えを出すためのニューロンを追加
#? activation="linear"というのは、ニューロンがどのように反応するかを決めています。
#? 恒等関数というのは、入ってきた情報をそのまま出すというシンプルなルール
model.add(Dense(n_out, activation="linear")) # 活性化関数に恒等関数

# ? 「compile」は、ゲームのルールを決めるところ。二乗誤差は、間違えたときにどれだけ間違えたかを計算する方法。「SGD」というのは、間違いを直すための手順を決める方法
model.compile(loss="mean_squared_error", optimizer="sgd") # 損失関数に二乗誤差、最適化アルゴリズムにSGDを使用してコンパイル

# ? 最後に、作った頭脳がどんなものかを、一覧で見ることができる
print(model.summary())

# ! 学習。　構築したモデルを使って、学習を行います。学習にはモデルオブジェクトのfit()メソッドを使います。
# ? model.fit関数は、ニューラルネットワークモデルを訓練するために使用
# ? x: これは学習に使うデータです。例えば、写真や数字のリストなど、モデルが学習するための情報
# ? t: これは、学習データxに対する正解、または目標となる値です。モデルがxを見たときに予測すべき正しい答えです
# ? batch_size=batch_size: これは、一度に学習するデータの量を指します。例えばbatch_size=8なら、8個のデータごとにモデルを更新します。小さなステップで学習していくことになります
# ? epochs=2000: これは、学習データ全体を何回繰り返して学習させるかを指します。ここでは2000回繰り返して、同じデータで何度も学習させています。これにより、モデルはデータのパターンをより良く理解できるようになります
# ? validation_split=0.1: これは、全学習データのうち、どのくらいを検証用に取っておくかを指します。0.1は10%を意味していて、学習には使わずにモデルの性能をテストするために使います。これにより、モデルが新しいデータに対してどれくらい上手に予測できるかをチェックできます
history = model.fit(x, t, batch_size=batch_size, epochs=500, validation_split=0.1) #  # 10%のデータを検証用に使う

# ! 学習の遷移
# ? 学習過程での誤差（loss）と検証用データに対する誤差（val_loss）をhistoryオブジェクトから取り出し、それぞれ変数lossとvlossに格納
# ? lossはモデルが訓練データから学習する際にどれだけ誤っているか、val_lossは未知のデータ（検証用データ）に対するモデルの性能を示します
loss = history.history['loss'] # 訓練用データの誤差
vloss = history.history['val_loss'] # 検証用データの誤差

# ? np.arange(len(loss))は、エポック数に対応する数列を生成します。これにより、横軸にエポック数、縦軸に誤差の値をプロットする
plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(vloss)), vloss)
plt.show()

# ! 学習済みモデルの使用。　predict()メソッドにより、学習済みモデルを使用し予測を行うことができます。
plt.plot(x, model.predict(x)) ## モデルを使用し予測を行う
plt.plot(x, t)
plt.show()

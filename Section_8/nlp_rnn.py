"""RNNによる自然言語処理
        RNNを使って、文書の自動作成を行います。
        今回は、江戸川乱歩の「怪人二十面相」を学習データに使い、乱歩風の文章を自動生成します。
        文章における文字の並びを時系列データと捉えて、次の文字を予測するようにRNNを訓練します。"""

# !テキストデータの前処理
import re
with open("/Users/matsuzakidaiki/Desktop/python/nlp_bot/Section_8/kaijin_nijumenso.txt", mode="r", encoding="utf-8") as f:
    text_original = f.read()

text = re.sub("《[^》]+》", "", text_original) # ルビの削除
text = re.sub("［[^］]+］", "", text) # 読みの注意の削除
text = re.sub("[｜ 　]", "", text) # | と全角半角スペースの削除
print("文字数", len(text))  # len() で文字列の文字数も取得可能 output->文字数 110323
# print(text)

# !RNNの各設定
n_rnn = 10  # 時系列の数
batch_size = 128
epochs = 60
n_mid = 128  # 中間層のニューロン数

# !文字のベクトル化
# ?各文字をone-hot表現で表し、時系列の入力データおよび正解データを作成します。
# ?今回はRNNの最後の時刻の出力のみ利用するので、最後の出力に対応する正解のみ必要になります。
import numpy as np
# インデックスと文字で辞書を作成
chars = sorted(list(set(text)))# setで文字の重複をなくし、各文字をリストに格納する
# print(chars)
print("文字数（重複無し）", len(chars)) #output-> 文字数（重複無し） 1249
char_indices = {}  # 文字がキーでインデックスが値
# 以下の辞書は後ほど使う
for i, char in enumerate(chars):
    char_indices[char] = i
indices_char = {}  # インデックスがキーで文字が値
for i, char in enumerate(chars):
    indices_char[i] = char

#? 時系列データと、それから予測すべき文字を取り出します
time_chars = []
next_chars = []
for i in range(0, len(text) - n_rnn):
    time_chars.append(text[i: i + n_rnn])
    next_chars.append(text[i + n_rnn])
#? 入力と正解をone-hot表現で表します。入力xと正解tを作る。
x = np.zeros((len(time_chars), n_rnn, len(chars)), dtype=np.bool)
t = np.zeros((len(time_chars), len(chars)), dtype=np.bool)
for i, t_cs in enumerate(time_chars):
    t[i, char_indices[next_chars[i]]] = 1 # 正解をone-hot表現で表す
    for j, char in enumerate(t_cs):
        x[i, j, char_indices[char]] = 1 # 入力をone-hot表現で表す
# print("xの形状", x.shape)
# print("tの形状", t.shape)

# !モデルの構築
# ?Kerasを使ってRNNを構築します。今回も、SimpleRNN層を使います。
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
model = Sequential()
model.add(SimpleRNN(n_mid, input_shape=(n_rnn, len(chars))))
model.add(Dense(len(chars), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
print(model.summary())

# !文書生成用の関数
# ? 各エポックの終了後、文章を生成するための関数を記述します。
# ? LambdaCallbackを使って、エポック終了時に実行される関数を設定します。
from keras.callbacks import LambdaCallback
def on_epoch_end(epoch, logs):
    print("エポック：", epoch)
    beta = 5  # 確率分布を調整する定数
    prev_text = text[0:n_rnn]  # 入力に使う文字
    created_text = prev_text  # 生成されるテキスト
    print("シード: ", created_text)
    for i in range(400):
        # 入力をone-hot表現に
        x_pred = np.zeros((1, n_rnn, len(chars)))
        for j, char in enumerate(prev_text):
            x_pred[0, j, char_indices[char]] = 1
        # 予測を行い、次の文字を得る
        y = model.predict(x_pred)
        p_power = y[0] ** beta  # 確率分布の調整
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))
        next_char = indices_char[next_index]
        created_text += next_char
        prev_text = prev_text[1:] + next_char
    print(created_text)
    print()
# エポック終了後に実行される関数を設定
epock_end_callback= LambdaCallback(on_epoch_end=on_epoch_end)

# !学習
# ? 構築したモデルを使って、学習を行います。
# ? fit( )メソッドをではコールバックの設定をし、エポック終了後に関数が呼ばれるようにします。
# ? 学習には数十分程度かかるので、時間のない方はエポック数を少なくして実行しましょう。
history = model.fit(x, t,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[epock_end_callback])

# !学習の推移
# ?誤差の推移を確認します。
%matplotlib inline
import matplotlib.pyplot as plt

loss = history.history['loss']
plt.plot(np.arange(len(loss)), loss)
plt.show()



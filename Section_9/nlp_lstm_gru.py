import re

# !LSTM、GRUによる自然言語処理
# ?LSTM、GRUを使って、文書の自動作成を行います。
# ?今回も、江戸川乱歩の「怪人二十面相」を学習データに使い、乱歩風の文章を自動生成します。
# ?以前にSimpleRNNを扱った時と同じように、文章における文字の並びを時系列データと捉えて、次の文字を予測するように訓練します

# !テキストデータの前処理
# ?今回は、LSTM、GRU共に学習データとして青空文庫の「怪人二十面相」を使います。
# ?最初に、テキストデータに前処理を行います。
with open("/Users/matsuzakidaiki/Desktop/python/nlp_bot/Section_9/kaijin_nijumenso.txt", mode="r", encoding="utf-8") as f:
    text_original = f.read()

text = re.sub("《[^》]+》", "", text_original) # ルビの削除
text = re.sub("［[^］]+］", "", text) # 読みの注意の削除
text = re.sub("[｜ 　]", "", text) # | と全角半角スペースの削除
print("文字数", len(text))  # len() で文字列の文字数も取得可能


# !各設定
n_rnn = 10
batch_size = 128
epochs = 60
n_mid = 255

# !文字のベクトル化
# ?各文字をone-hot表現で表し、時系列の入力データおよび正解データを作成します。
# ?今回もRNNの最後の時刻の出力のみ利用するので、最後の出力に対応する正解のみ必要になります。
import numpy as np

# ?# インデックスと文字で辞書を作成
chars = sorted(list(set(text)))
print("文字数(重複なし)", len(chars))
char_indices = {} # 文字がキーでインデックスが値
for i, char in enumerate(chars):
    char_indices[char] = i
indices_char = {} # インデックスがキーで文字が値
for i, char in enumerate(chars):
    indices_char[i] = char

# ?# 時系列データと、それから予測すべき文字を取り出します
time_chars = []
next_chars = []
for i in range(0, len(text) - n_rnn):
    time_chars.append(text[i: i + n_rnn])
    next_chars.append(text[i + n_rnn])

#? 入力と正解をone-hot表現で表します
x = np.zeros((len(time_chars), n_rnn, len(chars)), dtype=np.bool)
t = np.zeros((len(time_chars), len(chars)), dtype=np.bool)
for i, t_cs in enumerate(time_chars):
    t[i, char_indices[next_chars[i]]] = 1  # 正解をone-hot表現で表す
    for j, char in enumerate(t_cs):
        x[i, j, char_indices[char]] = 1 # 入力をone-hot表現で表す

print("Xの形状", x.shape)
print("tの形状", t.shape)


# !LSTMモデルの構築
# ?Kerasを使ってLSTMを構築します。
from keras.models import Sequential
from keras.layers import Dense, LSTM

model_lstm = Sequential()
model_lstm.add(LSTM(n_mid, input_shape=(n_rnn, len(chars))))
model_lstm.add(Dense(len(chars), activation="softmax"))
model_lstm.compile(loss="categorical_crossentropy", optimizer="adam")
print(model_lstm.summary())

# !文書生成用の関数
# ?各エポックの終了後、文章を生成するための関数を記述します。
# ?LambdaCallbackを使って、エポック終了時に実行される関数を設定します。
from keras.callbacks import LambdaCallback

def on_epoch_end(epoch, logs):
    print("エポック：", epoch)
    
    beta = 5
    prev_text = text[0:n_rnn]
    created_text = prev_text
    
    print("シード：", created_text)
    
    for i in range(400):
        x_pred = np.zeros((1, n_rnn, len(chars)))
        for j, char in enumerate(prev_text):
            x_pred[0, j , char_indices[char]] = 1
        
        y = model.predict(x_pred)
        p_power = y[0] ** beta
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))
        next_char = indices_char[next_index]
        
        created_text += next_char
        prev_text = prev_text[1:] + next_char
    
    print(created_text)
    print()

epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)


# !学習
# ?構築したLSTMを使って、学習を行います。
# ?fit( )メソッドでコールバックの設定をし、エポック終了後に関数が呼ばれるようにします。
# ?学習には数時間かかるので、時間のない方はエポック数を少なくして実行しましょう。
model = model_lstm
history_lstm = model_lstm.fit(x, t, batch_size=batch_size, epochs=epochs, callbacks=[epoch_end_callback])

# !GRUモデルの構築
# ?Kerasを使ってGRUを構築します。
from keras.layers import GRU

model_gru = Sequential()
model_gru.add(GRU(n_mid, input_shape=(n_rnn, len(chars))))
model_gru.add(Dense(len(chars), activation="softmax"))
model_gru.compile(loss="categorical_crossentropy", optimizer="adam")
print(model_gru.summary())

# !学習
# ?構築したGRUを使って、学習を行います。
# ?fit( )メソッドでコールバックの設定をし、エポック終了後に関数が呼ばれるようにします。
# ?LSTMと同じく学習には数時間程度かかるので、時間のない方はエポック数を少なくして実行しましょう。
model = model_gru
history_gru = model_gru.fit(x, t, batch_size=batch_size, epochs=epochs, callbacks=[epoch_end_callback])


# !学習の推移
# ?誤差の推移を確認します。
import matplotlib.pyplot as plt

loss_lstm = history_lstm.history['loss']
loss_gru = history_gru.history['loss']

plt.plot(np.arange(len(loss_lstm)), loss_lstm, label="LSTM")
plt.plot(np.arange(len(loss_gru)), loss_gru, label="GRU")
plt.legend()
plt.show()


# !生成された文章を比較
"""LSTM、GRU、それぞれで生成された文章を比較してみましょう。

LSTM
Epoch 60/60
110313/110313 [==============================] - 126s 1ms/step - loss: 0.1493
エポック: 59
シード: そのころ、東京中の町
そのころ、東京中の町という町、家という家では、ふたり以上の人が顔をあわせさえすれば、まるでお天気のあいさつでもするように、怪人「二十面相」のうわさばかりしているというのも、じつは、こわばかくで、ぼくをひっててやけるかもですね。その部屋にはあきません。おしてお目をとりだっしましたけれど、賊は、あいつは邸内が三人の前まになにのをつかれたというのです。
それでも、外見らめてあるというからだ。手がありません。
「二十面相のやつ、今夜ですよ。手紙のんのは、小林少年の苦心に、子どおりたんださい。このあばかな声ですから、そのじゃにともの見れています。もしかむこんわけは、何かしこそこしだ。こぎだから、ばかりタとしておね。ぼくはそうですね。あれは何かわらのことを、かけつけているのでしょうか。またかと大き賊の部下があのがわらをじて、じて、ゆっくりむずねばながってもたじつめて、庭園のことへはそうかさえしまいました。
「ああ、よかっ

GRU
Epoch 60/60
110313/110313 [==============================] - 104s 942us/step - loss: 0.2000
エポック: 59
シード: そのころ、東京中の町
そのころ、東京中の町という町、家という家では、ふたり以上の人が顔をあわせさ。すると、その下から黒々とした頭があらわれました。つぎには、おとうさんにちょっと会われてください。ぼくは少しやらくるようにしますと、電燈がよいつけて、主人の壮二君と、赤井寅三に、「二十面相」たいには、十分かしたというのかね。」
「ええ、おくびょうのようですけれど、なんだかそんな気がするのです。」
「だが、そんなことはないかなければよりませんでした。
「ほかのものならばかまわない。ダイヤなぞお金さえ出せば手にこぎってはたかぬかまりませんだ。それをしゃべりっていって、目的をはたしてしまったのですから、むりもないことです。
「いや、なんとも申しあげようもありません。二十面相がこれほどの腕まえとは知りませんでした。相手がですよ。それが二十面相の部下に会ったのか。いったい、どこで？どうして？」
さすがの警官はといって、やぶやから手もにもおわしてや

今回のケースでは、GRUで生成された文章の方が自然に見えますね。"""
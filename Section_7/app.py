import re
import pickle
import matplotlib.pyplot as plt
from janome.tokenizer import Tokenizer
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


with open("/Users/matsuzakidaiki/Desktop/python/nlp_bot/Section_7/wagahaiwa_nekodearu.txt", mode="r", encoding="utf-8") as f:  # ファイルの読み込み
    wagahai_original = f.read()

wagahai = re.sub("《[^》]+》", "", wagahai_original)  # ルビの削除
wagahai = re.sub("［[^］]+］", "", wagahai)  # 読みの注意の削除
wagahai = re.sub("[｜ 　「」\n]", "", wagahai)  # | と全角半角スペース、「」と改行の削除

seperator = "。"  # 。をセパレータに指定
wagahai_list = wagahai.split(seperator)  # セパレーターを使って文章をリストに分割する
wagahai_list.pop()  # 最後の要素は空の文字列になるので、削除
wagahai_list = [x + seperator for x in wagahai_list]  # 文章の最後に。を追加

t = Tokenizer()

wagahai_words = []
for sentence in wagahai_list:
    wagahai_words.append(list(t.tokenize(sentence, wakati=True)))  # 文章ごとに単語に分割し、リストに格納

with open('wagahai_words.pickle', mode='wb') as f:  # pickleに保存
    pickle.dump(wagahai_words, f)

with open('wagahai_words.pickle', mode='rb') as f:
    wagahai_words = pickle.load(f)

# print(wagahai_words[:10])

model = word2vec.Word2Vec(wagahai_words,
                          vector_size=100,
                          min_count=5,
                          window=5,
                          epochs=20,
                          sg=0)
# print(model.wv.vectors.shape)  # 分散表現の形状
# print(model.wv.vectors)
# print(len(model.wv.index_to_key))
print(model.wv.__getitem__("猫"))
print(model.wv.most_similar("猫"))

a = model.wv.__getitem__("猫")
b = model.wv.__getitem__("人間")
cos_sim = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
print(cos_sim)
# print(model.wv.most_similar("名前"))
# c = model.wv.__getitem__("名前")
# d = model.wv.__getitem__("娘")
# cos_sim_2 = np.dot(c, d) / np.linalg.norm(c) / np.linalg.norm(d)
# print(cos_sim_2)

# model.wv.most_similar(positive=['人間', '猫'], negative=['夢'])
# model.wv.most_similar(positive=['教師'], negative=['夢'])

print(wagahai_words[:10])

tagged_documents = []
for i, sentence in enumerate(wagahai_words):
    tagged_documents.append(TaggedDocument(sentence, [i]))

model = Doc2Vec(documents=tagged_documents,
                vector_size=100,
                min_count=5,
                window=5,
                epochs=20,
                dm=0)

print(wagahai_words[0])
print(model.dv[0])
print(model.dv.most_similar(0))
# 類似文章を出力
for p in model.dv.most_similar(0):
    print(wagahai_words[p[0]])


x = np.linspace(-np.pi,np.pi).reshape(-1, 1)
t = np.sin(x)
plt.plot(x, t)
# plt.show()

batch_size = 8  # バッチサイズ
n_in = 1  # 入力層のニューロン数
n_mid = 20  # 中間層のニューロン数
n_out = 1  # 出力層のニューロン数

# 入力層、中間層、出力層の３層のニューラルネットワークを構築
model = Sequential()
model.add(Dense(n_mid, input_shape=(n_in,), activation="sigmoid"))  # 活性化関数にシグモイド関数
model.add(Dense(n_out, activation="linear"))  # 活性化関数に恒等関数
model.compile(loss="mean_squared_error", optimizer="sgd")  # 損失関数に二乗誤差、最適化アルゴリズムにSGDを使用してコンパイル
print(model.summary())

history = model.fit(x, t, batch_size=batch_size, epochs=2000, validation_split=0.1)

loss = history.history['loss']
vloss = history.history['val_loss']

plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(vloss)), vloss)
# plt.show()

plt.plot(x, model.predict(x))
plt.plot(x, t)
# plt.show()
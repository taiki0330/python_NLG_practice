"""対話コーパスの前処理
以降のレクチャーでは、Seq2Seqを使って対話文を生成します。
宮沢賢治の小説を学習データに使い、賢治風の返事ができるようにモデルを訓練します。
今回は、そのための準備として、コーパスに前処理を行います。
小規模なコーパスでも対話文が生成ができるように、漢字は全てひらがなに変換します。"""

"""テキストデータの前処理
今回は、宮沢賢治の小説「銀河鉄道の夜」「セロ弾きのゴーシュ」「注文の多い料理店」などをコーパスとします。
コーパスが大きほど精度が高まるのですが、学習に時間がかかり過ぎても本コースに支障があるので、10作品に抑えておきます。"""

import re #正規表現のためreをインポート

novels = ["gingatetsudono_yoru.txt", "serohikino_goshu.txt", "chumonno_oi_ryoriten.txt",
         "gusukobudorino_denki.txt", "kaeruno_gomugutsu.txt", "kaino_hi.txt", "kashiwabayashino_yoru.txt",
         "kazeno_matasaburo.txt", "kiirono_tomato.txt", "oinomorito_zarumori.txt"]  # 青空文庫より


text = ""
for novel in novels:
    with open("/Users/matsuzakidaiki/Desktop/python/nlp_bot/Section_10/kenji_novels/"+novel, mode="r", encoding="utf-8") as f:  # ファイルの読み込み
        text_novel = f.read()
    text_novel = re.sub("《[^》]+》", "", text_novel)  # ルビの削除
    text_novel = re.sub("［[^］]+］", "", text_novel)  # 読みの注意の削除
    text_novel = re.sub("〔[^〕]+〕", "", text_novel)  # 読みの注意の削除
    text_novel = re.sub("[ 　\n「」『』（）｜※＊…]", "", text_novel)  # 全角半角スペース、改行、その他記号の削除
    text += text_novel

print("文字数:", len(text))
print(text)

from pykakasi import kakasi

seperator = "。"  # 。をセパレータに指定
sentence_list = text.split(seperator)  # セパレーターを使って文章をリストに分割する
sentence_list.pop() # 最後の要素は空の文字列になるので、削除
sentence_list = [x+seperator for x in sentence_list]  # 文章の最後に。を追加

kakasi = kakasi()
kakasi.setMode("J", "H")  # J(漢字) からH(ひらがな)へ
conv = kakasi.getConverter()
for sentence in sentence_list:
    print(sentence)
    print(conv.do(sentence))
    print()

# エラーが発生しました。
# pykakasiの辞書に無い「苹」という文字が問題となっているようです。
# この文字を予めひらがなに変換した上で、再び変換を行います。
text = text.replace("苹果", "ひょうか")

seperator = "。"
sentence_list = text.split(seperator) 
sentence_list.pop() 
sentence_list = [x+seperator for x in sentence_list]

for sentence in sentence_list:
    print(sentence)
    print(conv.do(sentence))
    print()

# エラーは発生していませんね。
# 問題なくひらがなに変換できたようです。

"""
次に、set()を使って文字の重複を無くし、使用されている文字の一覧を表示してみましょう。"""

kana_text = conv.do(text)  # 全体をひらがなに変換
print(set(kana_text))  # set()で文字の重複をなくす

# ひらがなとカタカナ、一部の記号のみが残りました。


"""テキストデータの保存¶
テキストデータを保存し、いつでも使えるようにします。"""
print(kana_text)
with open("kana_kenji.txt", mode="w", encoding="utf-8") as f:
    f.write(kana_text)


"""保存したテキストファイルを読み込み、保存できていることを確認します。"""
with open("kana_kenji.txt", mode="r", encoding="utf-8") as f:
    kana_text = f.read()
print(kana_text)


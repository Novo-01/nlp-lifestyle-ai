# src/generate_artist_csv.py

import csv
import os

# 保存先
os.makedirs("data/raw", exist_ok=True)
filename = "data/raw/artist.csv"

# アーティストっぽい文章データ
texts = [
    "今日の空、絵の具を垂らしたみたいだった。世界がキャンバスに見える瞬間。",
    "絵が完成した瞬間、いつも少しだけ自分と対話できた気がする。不思議な感覚。",
    "「美しい」は技術じゃなくて、視点だと思うんだよね。どこに心が動いたか、がすべて。",
    "アトリエでコーヒー飲みながら作業してたら、BGMのピアノと筆の音が溶け合ってた。",
    "カラーパレットがごちゃごちゃしてる日ほど、感情がうまく出てる気がする。",
    "描いてる途中で迷子になった。でもその迷子こそがアートなんだと思うようになった。",
    "SNSに上げる用の作品じゃなくて、自分にしかわからない絵を描いてみた。たまには大事。",
    "「伝えたいこと」がないときほど、絵が静かに語りだす。不思議だけど、ほんとにある。",
    "色に正解なんてない。今日の私は、青に怒りを込めた。",
    "アートって自己表現じゃなくて、自己発見かもしれない。描いて初めて気づく感情がある。"
]

# CSV書き込み
with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    for text in texts:
        writer.writerow([text, "artist"])

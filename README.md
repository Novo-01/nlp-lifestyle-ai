# nlp-lifestyle-ai

# 🧠 語彙の生態系AI：言葉からライフスタイルを診断するアプリ

## 🎯 コンセプト

SNS投稿や日記のような文章から「どんな語彙を使っているか？」をもとに、  
その人のライフスタイル傾向（クリエイター型？リア充型？タスク管理系？など）を推定するWebアプリです。

「語彙から生き方を想像する」── NLP × デザイン思考のちょっとユニークな診断体験を実現しました。

---

## 🚀 デモ（スクリーンショット）

※ `streamlit run app.py` でローカル実行可能  
（希望があればStreamlit Cloud等で公開もOK）

![demo](./path/to/screenshot.png)

---

## ⚙️ 技術構成

| 分類 | 使用技術 |
|------|----------|
| 自然言語処理 | spaCy（ja_ginza） |
| 特徴量変換   | TF-IDF（sklearn） |
| クラスタリング | KMeans |
| フロントエンド | Streamlit |
| データ管理 | pandas / joblib |

---

## 🧪 分析ステップ

1. 日本語投稿文を形態素解析し、名詞・動詞・形容詞のみ抽出（GiNZA）
2. TF-IDFで文章をベクトル化
3. KMeansクラスタリングでライフスタイル分類
4. クラスタ番号と使用語彙を診断結果として表示

---

## 📁 ディレクトリ構成

nlp-lifestyle-ai/
├── app.py                  # Streamlit診断アプリ本体
├── src/
│   └── utils.py           # トークン抽出関数（GiNZA）
├── notebooks/             # 分析用Jupyterノートブック
│   └── test.ipynb
├── models/
│   ├── vectorizer.joblib  # TF-IDFモデル
│   └── kmeans_model.joblib # クラスタリングモデル
├── data/
│   └── sample_users.csv
├── requirements.txt
└── README.md

---

## 📦 実行方法

# ライブラリインストール
pip install -r requirements.txt

# Streamlitアプリ起動
streamlit run app.py


from src.utils import extract_keywords
import streamlit as st
import joblib
import spacy

# ---------------------------
# モデルの読み込み
# ---------------------------
vectorizer = joblib.load("notebooks/models/vectorizer.joblib")
kmeans = joblib.load("notebooks/models/kmeans_model.joblib")
nlp = spacy.load("ja_ginza")

# ---------------------------
# ユーザー入力
# ---------------------------
st.title("🧠 あなたの語彙からライフスタイルを診断！")
text = st.text_area("あなたの最近のSNS投稿や日記っぽい文章を貼ってください")

# ---------------------------
# 解析処理
# ---------------------------
if text:
    tokens = extract_keywords(text)  # ← utilsから関数を呼び出し
    joined = " ".join(tokens)
    vec = vectorizer.transform([joined])
    cluster = kmeans.predict(vec)[0]

    st.subheader("🧬 あなたの診断結果")
    st.markdown(f"### ✅ クラスタ番号：`{cluster}`")
    st.write("※ クラスタごとの特徴づけは今後追加予定！")

    st.subheader("🔍 使用された語彙（トークン）")
    st.write(tokens)

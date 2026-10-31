# utils/data_processor.py

import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
from huggingface_hub import login
import os

# Hugging Face Hub にログイン (モデルダウンロード前に実行)
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("INFO: Logged into Hugging Face Hub successfully.")
    except Exception as e:
        print(f"WARNING: Failed to log into Hugging Face Hub: {e}")


# --- グローバル変数とモデルの初期化 ---

# Janome トークナイザーをグローバルに初期化 (関数外で一度だけ実行)
t = Tokenizer()

# ... login ...
@st.cache_resource
def load_embedding_model():
    model_name = 'intfloat/multilingual-e5-large'
    try:
        # モデルのロードを試みる
        return SentenceTransformer(model_name, token=hf_token)
    except Exception as e:
        # ロードに失敗した場合、エラーメッセージを表示して None を返す
        st.error(f"埋め込みモデルのダウンロードに失敗しました。クラスタリング機能は利用できません。エラー: {e}")
        return None # 失敗時には None を返す
embedding_model = load_embedding_model() # <-- ここにいるか確認！





# ストップワードのリストを定義 (グローバル変数)
STOP_WORDS = [
    'さん', 'こと', 'もの', 'よう', 'そう', 'ため', 'ちゃん', 'くん', 'って', '的',
    'みたい', 'の', 'です', 'ます', 'ね', 'よ', 'だ', 'です', 'ます', 'か', 'な',
    'これ', 'あれ', 'それ', 'どの', 'この', 'その',
    'なる', 'られる', 'れる', 'ある', 'いる', 'する', 'いう', 'れる', 'できる',
    'ない', 'ません', 'ありません', 'です', 'ませ', 'たら', 'れば', 'から', 'まで', 'より',
    'で', 'に', 'を', 'が', 'と', 'は', 'も', 'へ', 'と', 'ば', 'なら',
    'し', 'し', 'ん', 'て', 'よ', 'ね', 'な', 'だ', 'い', 'ん',
    '的', '性', '化', 'さ', 'み', 'め', 'さ', 'さ', 'そう',
    '笑い', 'ほんと', 'けど', 'けどね', 'けどな', 'だけど', 'だから',
    'います', 'でした', 'ました', 'さま', '様', 'みんな', '人', '方', '私', '僕', '俺', '彼女', '彼',
]


# --- 関数定義 ---

def extract_top_comments(df: pd.DataFrame, label: str, top_n: int = 5) -> list:
    """
    データフレームから指定された感情ラベルのコメントを抽出し、確信度が高い順に並べて返す。
    """
    if label not in df["感情"].unique():
        return []
    filtered_df = df[df["感情"] == label].sort_values(by="確信度", ascending=False)
    top_comments = filtered_df["コメント"].head(top_n).tolist()
    return top_comments

def get_top_words(comments_text: list, top_n: int = 10) -> list:
    """
    コメントのリストから名詞を抽出し、出現頻度が高い順にランキングを返す。
    ストップワードを除外する。
    """
    # グローバルな Janome トークナイザー 't' を使用
    words = []
    for token in t.tokenize(" ".join(comments_text)):
        if token.part_of_speech.startswith('名詞'):
            base_form = token.base_form
            if base_form not in STOP_WORDS and len(base_form) > 1:
                words.append(base_form)
    
    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)
    return top_words


def create_wordcloud(comments_text: list, background_color="white", width=800, height=400) -> str:
    """
    コメントのリストからワードクラウド画像を生成し、Base64エンコードされた文字列として返す。
    """
    if not comments_text:
        return ""
        
    font_path = "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc" # macOSのデフォルトフォント
    if not os.path.exists(font_path):
        print(f"WARNING: 日本語フォントが見つかりません({font_path})。ワードクラウドの日本語表示が崩れる可能性があります。")
        font_path = None
    processed_text = " ".join([token.base_form for token in t.tokenize(" ".join(comments_text)) if token.part_of_speech.startswith('名詞') and token.base_form not in STOP_WORDS and len(token.base_form) > 1])

    if not processed_text:
        return ""

    wc = WordCloud(
        font_path=font_path, background_color=background_color, width=width, height=height,
        max_words=100, collocations=False
    )
    wc.generate(processed_text)

    buf = io.BytesIO()
    fig = plt.figure(figsize=(width/100, height/100))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def cluster_comments(comments: list, n_clusters: int = 5):
    """
    コメントをトピックでクラスタリングし、各クラスターの代表ワードを抽出する。
    """
        # --- デバッグ追加 ---
    print("DEBUG: cluster_comments function started.")
    print(f"DEBUG: Number of comments for clustering: {len(comments)}")
        # --- ここにチェックを追加 ---
    if embedding_model is None:
        print("DEBUG: embedding_model is None.") # デバッグ追加
        return "クラスタリングモデルのロードに失敗しました。"

    if len(comments) < n_clusters or len(comments) < 20:
        print(f"DEBUG: Not enough comments ({len(comments)}) for clustering.") # デバッグ追加
        return "コメント数が少なすぎます。クラスタリングには最低でも20件以上のコメントが必要です。"
    
    # 1. テキストをベクトルに変換（エンベディング）
    print("DEBUG: Starting text embedding...") # デバッグ追加
    comment_embeddings = embedding_model.encode(comments)
    print("DEBUG: Text embedding completed.") # デバッグ追加
    
    # 2. KMeansクラスタリングを実行
    print("DEBUG: Starting KMeans clustering...") # デバッグ追加
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(comment_embeddings)
    cluster_labels = kmeans.labels_
    print("DEBUG: KMeans clustering completed.") # デバッグ追加
    
    clustered_data = pd.DataFrame({'comment': comments, 'cluster': cluster_labels})
    cluster_topics = {}
    
    for cluster_id in range(n_clusters):
        cluster_comments = clustered_data[clustered_data['cluster'] == cluster_id]['comment'].tolist()
        top_words_in_cluster = get_top_words(cluster_comments, top_n=5)
        
        if cluster_comments:
            representative_comment = cluster_comments[0]
        else:
            representative_comment = "コメントなし"
        
        cluster_topics[cluster_id] = {
            'comments_count': len(cluster_comments),
            'top_words': [word for word, count in top_words_in_cluster],
            'representative_comment': representative_comment,
        }

    print("DEBUG: cluster_comments function finished and returning.") # デバッグ追加   
    return cluster_topics, cluster_labels



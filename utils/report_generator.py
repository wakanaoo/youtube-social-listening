import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

# --- ここに日本語フォント設定を追加 ---
# 日本語フォントのパスを指定
# macOSの場合の例
font_path_mac = '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc'
# Windowsの場合の例 (Meiryo UI, YuGothic など)
font_path_win = 'C:\\Windows\\Fonts\\meiryo.ttc'
# Linuxの場合の例 (Noto Sans CJK JP など)
font_path_linux = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

# OSに合わせて適切なフォントパスを設定
import platform
os_name = platform.system()
if os_name == 'Darwin': # macOS
    font_path = font_path_mac
elif os_name == 'Windows':
    font_path = font_path_win
else: # Linux
    font_path = font_path_linux

# フォントプロパティを設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic ProN', 'Meiryo', 'Yu Gothic', 'DejaVu Sans'] # フォントの候補を指定
plt.rcParams['font.size'] = 12

# 既存のフォント設定を上書き
# （WordCloudのfont_pathとは別に設定が必要です）
import os
import matplotlib.font_manager as fm
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()] + plt.rcParams['font.sans-serif']
else:
    print(f"WARNING: 日本語フォントが見つかりません({font_path})。グラフの日本語表示が崩れる可能性があります。")

# --- フォント設定ここまで ---

def plot_sentiment_pie(sentiment_results):
    """
    感情分析の結果を基に円グラフを生成し、Base64エンコードされた画像データを返す。

    Args:
        sentiment_results (list): analyze_sentiment_local から返されるような感情分析結果のリスト。
                                  例: [{'label': 'POSITIVE', 'score': 0.99}, ...]
    Returns:
        str: Base64エンコードされたPNG画像データ（HTMLの<img>タグで表示可能）。
    """
    # 感情ラベルごとのスコアを集計
    sentiment_counts = {}
    for result in sentiment_results:
        label = result['感情']
        # スコアをそのまま使うのではなく、単にカウントを増やす
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

    # 各感情のパーセンテージを計算
    total_count = sum(sentiment_counts.values())
    if total_count == 0:
        return "" # データがない場合は空文字列を返す

    labels = list(sentiment_counts.keys())
    sizes = [count / total_count for count in sentiment_counts.values()]

    # 円グラフの色を定義（オプション）
    colors = {
        'POSITIVE': 'lightgreen',
        'NEGATIVE': 'lightcoral',
        'NEUTRAL': 'lightgray'
    }
    # 定義されていないラベルの色はデフォルトで割り当てる
    pie_colors = [colors.get(label, 'lightblue') for label in labels]

    # 円グラフの作成
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    ax1.axis('equal')  # 円を真円に保つ

    # グラフを画像データとしてメモリに保存し、Base64エンコードする
    buf = io.BytesIO()
    fig1.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig1) # メモリを解放
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# --- ここから棒グラフ生成関数の追加 ---
def plot_sentiment_bar(sentiment_results):
    """
    感情分析の結果を基に棒グラフを生成し、Base64エンコードされた画像データを返す。
    """
    # 感情ラベルごとの数を集計
    sentiment_counts = {}
    for result in sentiment_results:
        label = result['感情'] # '感情' キーを使用
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

    # DataFrame に変換してプロット
    df_counts = pd.DataFrame(sentiment_counts.items(), columns=['感情', 'コメント数']).sort_values('コメント数', ascending=False)
    
    if df_counts.empty:
        return "" # データがない場合は空文字列を返す

    # グラフの色を定義（円グラフと合わせる）
    colors = {
        'POSITIVE': 'lightgreen',
        'NEGATIVE': 'lightcoral',
        'NEUTRAL': 'lightgray'
    }

    # グラフの作成
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 棒グラフを描画
    bars = ax.bar(df_counts['感情'], df_counts['コメント数'], color=[colors.get(label, 'lightblue') for label in df_counts['感情']])

    # 軸ラベルとタイトル
    ax.set_xlabel('感情ラベル', fontsize=12)
    ax.set_ylabel('コメント数', fontsize=12)
    ax.set_title('コメントの感情別件数', fontsize=16)
    
    # 棒の上にコメント数を表示
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=10)

    # グラフを画像データとしてメモリに保存し、Base64エンコードする
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


# --- ここから頻出単語の棒グラフ生成関数を追加 ---

def plot_word_frequency_bar(top_words: list, title: str, color: str = 'skyblue'):
    """
    頻出単語のランキングを棒グラフで生成し、Base64エンコードされた画像データを返す。

    Args:
        top_words (list): get_top_words 関数から返される (単語, 件数) のタプルのリスト。
        title (str): グラフのタイトル。
        color (str): 棒グラフの色。
    Returns:
        str: Base64エンコードされたPNG画像データ。
    """
    if not top_words:
        return "" # データがない場合は空文字列を返す

    # DataFrame に変換してプロット
    df_words = pd.DataFrame(top_words, columns=['単語', '件数'])
    
    # グラフの作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 棒グラフを描画（単語の順番を逆にして、件数が多いものが上にくるように）
    y_pos = range(len(df_words))
    ax.barh(y_pos, df_words['件数'], align='center', color=color)
    
    # Y軸のラベルを単語名にする
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_words['単語'])
    
    # 軸ラベルとタイトル
    ax.set_xlabel('件数', fontsize=12)
    ax.set_title(title, fontsize=16)
    
    # X軸の目盛りを整数にする
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # グラフを画像データとしてメモリに保存し、Base64エンコードする
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# --- ここから確信度分布グラフ生成関数の追加 ---

def plot_confidence_histogram(df: pd.DataFrame):
    """
    感情ラベルごとの確信度（スコア）のヒストグラムを生成し、Base64で返す。
    """
    if df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))

    # 感情ラベルごとにデータをフィルタリング
    positive_scores = df[df['感情'] == 'POSITIVE']['確信度'] / 100
    negative_scores = df[df['感情'] == 'NEGATIVE']['確信度'] / 100
    neutral_scores = df[df['感情'] == 'NEUTRAL']['確信度'] / 100

    # ヒストグラムを描画
    ax.hist(positive_scores, bins=10, alpha=0.6, label='ポジティブ', color='lightgreen')
    ax.hist(negative_scores, bins=10, alpha=0.6, label='ネガティブ', color='lightcoral')
    ax.hist(neutral_scores, bins=10, alpha=0.6, label='ニュートラル', color='lightgray')

    # 軸ラベルとタイトル
    ax.set_title('モデルの確信度分布', fontsize=16)
    ax.set_xlabel('確信度スコア (0.0 - 1.0)', fontsize=12)
    ax.set_ylabel('コメント数', fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # グラフを画像データとしてメモリに保存
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"
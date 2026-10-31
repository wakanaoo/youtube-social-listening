# services/sentiment_analyzer.py

from transformers import pipeline
from dotenv import load_dotenv
import os
import streamlit as st
import re
from janome.tokenizer import Tokenizer

# Janome トークナイザーをグローバルに初期化 (一度だけ)
t = Tokenizer()

# .env ファイルから環境変数を読み込む
load_dotenv()

# HUGGINGFACE のトークンを取得(未設定でも、公開モデルなら認証なしで動作する)
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or None

if not hf_token:
    print("WARNING: HUGGINGFACE_HUB_TOKEN is not set. Proceeding without authentication (OK for public models).")

# モデルのキャッシュディレクトリを設定
cache_dir = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

@st.cache_resource
def load_sentiment_analyzer(token: str):
    model_name = "koheiduck/bert-japanese-finetuned-sentiment" # 現在動作しているモデル

    analyzer = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        framework="pt",
        cache_dir=cache_dir,
        truncation=True,
        max_length=512,
        token=token
    )
    return analyzer

analyzer = load_sentiment_analyzer(hf_token)



# ファンのコメントや動画へのコメントで使われる単語（拡張版）
FAN_NOISE_WORDS = [
    '可愛い', 'かっこいい', '最高', '大好き', '尊い', '推し', '神', '天才', 'やばい',
    'わかる', 'それな', 'ほんと', 'うん', # 共感
    'ありがとう', 'いつも', '参考', 'ためになる', # 配信者への感謝/感想
    'すごい', '素晴らしい', 'うまい', 'さすが', # 感嘆
    '動画', '編集', '撮影', '見やすい', # 動画へのコメント
]

# 感情分析のためのテキスト前処理を行う関数
def preprocess_text_for_sentiment(text: str) -> str:
    """
    感情分析のためのテキスト前処理を行う。
    絵文字、顔文字、ネットスラングなどの一部を正規化・変換する。
    """
    processed_text = text

    # 1. 絵文字の感情語への変換（一般的なものを厳選）
    emoji_map = {
        "😂": "笑い", "🤣": "爆笑", "😊": "笑顔", "😄": "嬉しい", "😁": "嬉しい",
        "😀": "嬉しい", "😆": "嬉しい", "😇": "天使", "😍": "愛しい", "🥰": "大好き",
        "👍": "良い", "👎": "悪い", "❤️": "ハート", "💖": "ハート", "💙": "ハート",
        "🌟": "素晴らしい", "✨": "きらきら", "💯": "完璧", "🎉": "おめでとう", "🥳": "おめでとう",
        "🙏": "感謝", "🙌": "拍手", "😡": "怒り", "😠": "怒り", "😤": "不満",
        "😞": "がっかり", "😥": "悲しい", "😢": "悲しい", "😭": "泣く", "😱": "驚き",
        "🤔": "考える", "🙄": "呆れる", "😫": "疲れた", "😩": "疲れた", "😑": "無表情",
        "😌": "安心", "😉": "ウインク", "😏": "ニヤリ", "😗": "キス", "😙": "キス",
        "💋": "キス", "💔": "悲しみ", "💖": "喜び", "🤢": "気持ち悪い", "🤮": "吐き気",
        "😴": "眠い", "😮‍💨": "ため息", "😅": "汗",
    }
    for emoji, meaning in emoji_map.items():
        processed_text = processed_text.replace(emoji, f" {meaning} ")

    # 2. 顔文字の感情語への変換（簡易的）
    kaomoji_map = {
        "(^o^)／": "嬉しい", "(^o^)": "嬉しい", "(*^o^*)": "嬉しい",
        "(´・ω・`)": "悲しい", "(つд｀;)": "悲しい", "(;ω;)": "悲しい",
        "(>_<;)": "困惑", "( ´_ゝ`)": "冷たい", "(°Д°)": "驚き",
        "(￣ー￣)": "ふむ", "(≧∇≦)": "やった", "(^ ^;)": "汗",
        "( ´∀｀)": "笑顔", "( ´•ω•` )": "悲しい", "(´・ω・｀;)": "悲しい", # 少し修正
        "(;´Д｀)": "辛い", "(´Д｀)": "嫌だ", "( `皿´)": "怒り",
        "( `o´)": "怒り", "(・ω・)": "無表情", "( ´-`)": "考える",
        "(°_°)": "困惑", "(￣▽￣)": "満足", "(*´艸`*)": "笑い",
        "_(:з」∠)_": "疲れた", "(*^▽^*)": "嬉しい", "(((o(*ﾟ▽ﾟ*)o)))": "興奮",
        "(*´∇｀*)": "嬉しい", "(｀・ω・´)": "キリッ", "( ･´ｰ･｀)": "クール",
        "（＾ω＾）": "笑顔", "（´・ω・｀）": "悲しい", "（；ω；）": "泣き",
        "（つд｀）": "悲しい",
    }
    for kaomoji_str, meaning in kaomoji_map.items(): # 変数名を kaomoji_str に変更
        # re.escape() を使って文字列を正規表現パターンとしてエスケープする！
        processed_text = re.sub(re.escape(kaomoji_str), f" {meaning} ", processed_text)

    # 3. ネットスラングの処理（簡易的）
    # 「草」「w」の連続は「笑い」に変換。皮肉な場合はモデルが判断に迷う可能性あり
    processed_text = re.sub(r'[wWＷＷｗｗ]+', ' 笑い ', processed_text) # wwww -> 笑い
    processed_text = processed_text.replace('草', ' 笑い ') # 草 -> 笑い
    processed_text = processed_text.replace('乙', ' 乙 ') # お疲れ様、中立的なことが多い
    processed_text = processed_text.replace('まじ', ' 本当に ') # まじ -> 本当に (強さを示す)
    processed_text = processed_text.replace('ガチ', ' 本当に ') # ガチ -> 本当に (強さを示す)
    processed_text = processed_text.replace('エモい', ' 感動的 ')
    processed_text = processed_text.replace('尊い', ' 尊い ')
    processed_text = processed_text.replace('ヤバい', ' 素晴らしい ') #ポジティブ寄りに変換（文脈次第でネガティブにもなるが簡易化）

    # 4. 感嘆符・疑問符の連続を正規化 (強調を示す)
    processed_text = re.sub(r'！+', '！', processed_text)
    processed_text = re.sub(r'\!+', '！', processed_text)
    processed_text = re.sub(r'？+', '？', processed_text)
    processed_text = re.sub(r'\?+', '？', processed_text)
    
    # URLの削除 (感情分析には不要なため)
    processed_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', processed_text)
    
    # 余分な空白を削除
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()

    # 5. ファンのコメントなど、分析のノイズになりうるコメントを除外
    # コメントがこれらの単語のみで構成されているかチェック
    # 名詞を抽出して、その中にノイズ単語が多いか確認する
    t = Tokenizer()
    words = [token.base_form for token in t.tokenize(processed_text) if token.part_of_speech.startswith('名詞')]
    
    # コメント内の名詞の50%以上がノイズ単語なら、そのコメントを空にする
    noise_word_count = sum(1 for word in words if word in FAN_NOISE_WORDS)
    if len(words) > 0 and (noise_word_count / len(words)) > 0.5:
        return "" # コメントを空文字列にして分析対象から除外

    return processed_text

def analyze_sentiment_local(text: str):
    """
    Hugging Faceのローカルモデルを使用してテキストの感情分析を行います。
    分析前にテキストの前処理を行います。
    """
    preprocessed_text = preprocess_text_for_sentiment(text)
    
    # 前処理後のテキストが空になる場合があるためチェック
    if not preprocessed_text:
        return [{'label': 'NEUTRAL', 'score': 1.0}] # 空テキストはニュートラルと判断

    # print(f"DEBUG: Original Text: {text}") # デバッグ用
    # print(f"DEBUG: Preprocessed Text: {preprocessed_text}") # デバッグ用

    result = analyzer(preprocessed_text)
    return result
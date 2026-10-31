# services/llm_insights.py
import google.generativeai as genai
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@st.cache_data(show_spinner=False)
def generate_insight_with_llm(sentiment_df: pd.DataFrame, sentiment_label: str, keyword: str) -> str:
    """
    特定の感情のコメントをLLMで要約し、洞察を生成する。
    """
    # 1. 感情分析結果のDataFrameから、指定された感情のコメントだけを抽出
    filtered_comments = sentiment_df[sentiment_df["感情"] == sentiment_label]["コメント"].tolist()

    if not filtered_comments:
        return f"😔 「{sentiment_label}」に分類されたコメントが見つからなかったよ。分析するためのデータが足りないみたい…"

    # 2. 感情ラベルに応じてプロンプトを切り替える
    if sentiment_label == "NEGATIVE":
        analysis_points = """
        1. **主な不満点や課題**：
           - これらのコメントから読み取れる、最も重要な不満点や具体的な課題（例: **使用感、方法の難しさ、仕上がり、持続性、相性**など）を、箇条書きで3点挙げてください。
           - **注意:** もしコメントの内容がポジティブな皮肉やギャグであるなど、ネガティブに見えない表現が含まれている場合は、その旨を正直に指摘してください。

        2. **具体的な改善提案**：
           - 上記の不満点を解決するために、どのような改善が考えられますか？具体的な提案を3つ、箇条書きで提案してください。
        """
        summary_title = "ネガティブコメント全体から読み取れる総合的な印象を100字程度でまとめてください。"
    
    elif sentiment_label == "POSITIVE":
        analysis_points = """
        1. **主な評価点や魅力**：
           - これらのコメントから読み取れる、ユーザーが最も評価している点や具体的な魅力（例: **仕上がり、使いやすさ、持続力、デザイン、表現方法**など）を、箇条書きで3点挙げてください。
           - **注意:** もしコメントの内容がネガティブな皮肉であるなど、ポジティブに見えない表現が含まれている場合は、その旨を正直に指摘してください。

        2. **製品やサービスの強み**：
           - 上記の評価点から、このメイク方法や製品のどのような点が強みになっているか、具体的な観点で3つ、箇条書きで提案してください。
        """
        summary_title = "ポジティブコメント全体から読み取れる総合的な印象を100字程度でまとめてください。"

    else: # NEUTRAL やその他のラベル
        analysis_points = f"""
        1. **主な言及内容**：
           - これらのコメントで「{keyword}」がどのような文脈で使われているか、簡潔に箇条書きで3点挙げてください。
        """
        summary_title = "ニュートラルコメント全体から読み取れる総合的な印象を100字程度でまとめてください。"

    # 3. LLMに渡すプロンプトを作成
    comments_str = '\n'.join([f"- {c}" for c in filtered_comments])

    prompt = f"""
    あなたは**美容トレンドを分析する専門家**です。
    以下は、あるメイク方法に対するユーザーからのコメントです。

    【コメント一覧】
    {comments_str}

    【分析と提案】
    {analysis_points}

    3. **総合的な要約**：
       - {summary_title}

    ---
    **注意：**
    - 回答は上記の【分析と改善提案】の構成に沿って、日本語で分かりやすく記述してください。
    - コメントに具体的な情報が少ない場合は、その旨を正直に記載してください。
    - 不要な前置きや挨拶は不要です。
    """

    # 3. LLM APIを呼び出す
    try:
        model = genai.GenerativeModel('gemini-3.8-flash') # BedrockのAmazon Titanではないことに注意
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # APIキーが設定されていない場合や、API呼び出しに失敗した場合のハンドリング
        print(f"Error calling LLM API: {e}")
        return "😔 ごめんね、AIによる分析に失敗しちゃったみたい...。少し時間を置いてから、もう一度試してみてね！"
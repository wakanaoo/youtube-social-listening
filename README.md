# 💄 流行りのメイクを深掘り！コメント分析ツール

YouTubeのコメントを収集・分析し、特定のメイクや美容トレンドに対するユーザーの生の感想・評価を可視化するソーシャルリスニングツールです。

## 概要

ユーザーが入力したキーワードに関連するYouTube動画を検索し、コメントを自動収集。日本語BERTモデルによる感情分析、クラスタリング、LLMによる洞察生成を通じて、コメントの背後にあるニーズや不満を可視化します。

## 主要機能

- **キーワード検索・コメント自動取得**：YouTube動画をキーワード検索し、選択した動画のコメントを取得
- **感情分析**：日本語BERTモデルでコメントをポジティブ/ネガティブ/ニュートラルに分類
- **多角的な可視化**：感情割合グラフ、確信度フィルター、頻出単語ランキング、ワードクラウド
- **コメントの自動クラスタリング**：Sentence Transformers + KMeansで話題ごとにグループ化
- **LLMによる詳細分析**：Gemini APIでネガティブコメントから改善提案、ポジティブコメントから強みを生成
- **代表コメントの抽出・CSVダウンロード**

## 使用技術

- **フロントエンド/UI**: Streamlit
- **感情分析**: Transformers（`koheiduck/bert-japanese-finetuned-sentiment`）
- **クラスタリング**: Sentence Transformers（`intfloat/multilingual-e5-large`）, scikit-learn（KMeans）
- **LLM連携**: Google Gemini API（`gemini-3.8-flash`）
- **データ収集**: yt-dlp, youtube-search-python
- **前処理/可視化**: Janome, pandas, matplotlib, WordCloud

## ローカルでの起動方法

```bash
# 1. 仮想環境の作成・有効化
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 2. 依存パッケージのインストール
pip install -r requirements.txt

# 3. .env ファイルを作成し、Gemini APIキーを設定
echo "GEMINI_API_KEY=<発行したAPIキー>" > .env
# ※ 感情分析・クラスタリングのモデルは公開モデルのため、Hugging Faceトークンは不要です

# 4. アプリの起動
streamlit run app.py
```

## 更新履歴

2025年に開発したアプリをベースに、2026年にライブラリ更新・不具合修正・動作検証を実施しました。

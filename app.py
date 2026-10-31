# app.py

# app.py の冒頭に追記
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from services.sentiment_analyzer import analyze_sentiment_local
from services.youtube_collector import search_youtube_videos, get_comments
from datetime import datetime# datetimeをインポート
import time # <-- ここ！ time.sleep() を使うために、このモジュールをインポートする
from utils.report_generator import (plot_sentiment_pie, plot_sentiment_bar, plot_word_frequency_bar, 
    plot_confidence_histogram)
from utils.data_processor import extract_top_comments, create_wordcloud, get_top_words, cluster_comments  # ← ここに create_wordcloud があるか確認！  # ← 最初の import セクションに追加
from services.llm_insights import generate_insight_with_llm # 新しく追加

# ここにカスタムCSSを追加
st.markdown(
    """
    <style>
    /* アプリ全体のフォント、行間、文字サイズを調整 */
    html, body, [class*="st-"] {
        font-family: sans-serif; /* 美容系に合うフォントがあれば指定 */
        line-height: 1.6;
        font-size: 16px;
    }

    /* ボタンのスタイル */
    .stButton>button {
        background-color: #F28DA7; /* ボタンの背景色（テーマの primaryColor と合わせる） */
        color: white; /* ボタンのテキスト色 */
        border-radius: 25px; /* ボタンの角を丸く */
        border: none;
        padding: 10px 25px; /* パディングを調整 */
        font-size: 16px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2); /* 影を追加 */
        transition: all 0.3s ease; /* ホバー時のアニメーション */
    }
    .stButton>button:hover {
        background-color: #FFB3C6; /* ホバー時の色 */
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3); /* ホバー時の影を強調 */
        transform: translateY(-2px); /* わずかに上に移動 */
    }

    /* テキスト入力フィールド (st.text_input) のスタイル */
    .stTextInput>div>div>input {
        background-color: #FFFDFE; /* わずかに明るい背景 */
        border-radius: 15px; /* 丸み */
        border: 1px solid #FFE0EB; /* 細いボーダー */
        padding: 10px 15px;
        box-shadow: inset 1px 1px 3px rgba(0,0,0,0.1); /* 内側に影 */
    }

    /* セレクトボックス (st.selectbox) のスタイル */
    .stSelectbox>div>div {
        background-color: #FFFDFE;
        border-radius: 15px;
        border: 1px solid #FFE0EB;
        box-shadow: inset 1px 1px 3px rgba(0,0,0,0.1);
    }

    /* スピナー（例: 🔍 YouTube動画を探しているよ...）の色 */
    .stSpinner > div > div {
        border-top-color: #F28DA7 !important; /* スピナーの色をプライマリカラーに */
    }
    
    /* 成功メッセージ (st.success) のスタイル */
    .stSuccess {
        background-color: #E6FFE6; /* 薄い緑 */
        color: #006400; /* 濃い緑 */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    /* 警告メッセージ (st.warning) のスタイル */
    .stWarning {
        background-color: #FFFACD; /* 薄い黄 */
        color: #DAA520; /* 濃い黄 */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    /* サブヘッダー (st.subheader) のスタイル */
    h2 {
        color: #8C4773; /* 深いピンク/パープル系 */
        border-bottom: 2px solid #F28DA7; /* 下線 */
        padding-bottom: 5px;
        margin-top: 30px;
    }

    /* 円グラフやワードクラウドのキャプション */
    .stImage > figcaption {
        font-size: 1.1em;
        font-weight: bold;
        color: #4A4A4A;
        text-align: center;
        margin-top: 5px;
    }

    /* データフレームの見た目はStreamlitのデフォルトが良いことも多いが、もし調整するなら */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden; /* 角丸にするために必要 */
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- 3. アプリケーションのUIとロジック ---
st.markdown("""
# 💄 流行りのメイクを深掘り！
# コメント分析ツール
""")

# ユーザーのキーワード入力
keyword = st.text_input("✨ 気になる美容・メイクのキーワードを入力してね ✨ 例: 垢抜け メイク, 盛れる メイク")

# セッションステートの初期化
if 'analysis_data_ready' not in st.session_state:
    st.session_state.analysis_data_ready = False
    st.session_state.df = pd.DataFrame()
    st.session_state.analysis_data = None
    st.session_state.video_details = []
    st.session_state.keyword = ""
    st.session_state.clustering_run = False # <-- この行を追加！
    st.session_state.cluster_results = None # <-- この行も追加！

# キーワードが入力されたら動画検索を開始
if keyword:
    # キーワードが変わったらセッションステートをリセット
    if st.session_state.keyword != keyword:
        st.session_state.analysis_data_ready = False
        st.session_state.df = pd.DataFrame()
        st.session_state.video_details = []
        st.session_state.keyword = keyword
        st.session_state.all_comments = []
        st.session_state.clustering_run = False # <-- この行を追加！
        st.session_state.cluster_results = None # <-- この行も追加！
        
    with st.spinner("🔍 YouTube動画を探しているよ... ちょっと待ってね！"):
        video_results = search_youtube_videos(keyword)
    
    if video_results:
        # ユーザーが分析する動画をリストから手動で選択/解除
        selected_video = st.selectbox(
            "🎥 分析したい動画を1つ選んでね！",
            video_results
        )
        
        # コメント取得と分析を実行するボタン
        if st.button("🚀 この動画を分析開始！"):
            if selected_video:
                with st.spinner("💬 コメントを読み込み中だよ..."):
                    analysis_data = get_comments(selected_video)
                
                if analysis_data and 'comments' in analysis_data and analysis_data['comments']:
                    all_comments = analysis_data['comments']
                    st.session_state.analysis_data_ready = True
                    
                    # フィルタリングを削除し、全コメント数を表示
                    st.success(f"🎉 合計 {len(all_comments)} 件のコメントが見つかったよ！今から感情を分析するね...")

                    results = []
                    # フィルタリングせずに全コメントをループして感情分析
                    for comment in all_comments:
                        analysis_result = analyze_sentiment_local(comment)
                        if analysis_result and len(analysis_result) > 0:
                            label = analysis_result[0]['label']
                            score = analysis_result[0]['score']
                        else:
                            label = "UNKNOWN"
                            score = 0.0
                        results.append({"コメント": comment, "感情": label, "確信度": round(score * 100, 2)})
                        
                        # --- セッションステートに分析結果を保存 ---
                        st.session_state.df = pd.DataFrame(results)
                        st.session_state.analysis_data = analysis_data
                        st.session_state.video_details = [f"**{analysis_data.get('title', 'タイトル不明')}** (公開日: {analysis_data.get('upload_date', '日付不明')})"]

                else:
                    st.warning("🥲 ごめんね、この動画からはコメントが見つからなかったみたい...")
                    st.session_state.analysis_data_ready = False
            else:
                st.info("👆 分析を開始するには、動画を1つ選んでね！")
    else:
        st.warning("😥 キーワードに合う動画が見つからなかったよ。別の言葉で試してみてね！")


# --- 4. 分析結果の表示（セッションステートにデータがあれば常に表示） ---
if st.session_state.analysis_data_ready:
    df = st.session_state.df
    video_details = st.session_state.video_details
    analysis_data = st.session_state.analysis_data # analysis_data もセッションステートから取得

    st.subheader("📝 分析サマリー")
    st.info(f"📅 **分析対象動画:** \n" + "\n".join(video_details) + f"\n\n✨ **分析実施日:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.dataframe(df)

        # --- 全コメントの頻出単語ランキングの追加 ---
    st.subheader("🌍 全コメントの話題トレンド")
    
    # フィルタリング前の「all_comments」を使って分析
    if 'all_comments' in st.session_state and st.session_state.all_comments:
        all_words_list = get_top_words(st.session_state.all_comments)
        if all_words_list:
            st.write("#### 全コメントでよく出る単語")
            all_words_bar_base64 = plot_word_frequency_bar(all_words_list, "全コメントの頻出単語", color='gray')
            st.image(all_words_bar_base64)
            st.dataframe(pd.DataFrame(all_words_list, columns=["単語", "件数"]), hide_index=True)
        else:
            st.info("ごめんね！コメントが少ないから、全体の話題の傾向をつかむのが難しいみたい。もう少しデータがあれば頑張るよ！")
    # --- 追加ここまで ---

    st.subheader("🔍 分析の信頼性を調整")
    min_confidence = st.slider(
        "コメントを分析するAIの最低確信度スコアを選んでね",
        min_value=0.0,
        max_value=1.0,
        value=0.5, # 初期値は0.5
        step=0.05
    )

        # フィルターされた DataFrame を作成
    df_filtered = df[df['確信度'] >= (min_confidence * 100)] # 確信度は0-100なので、*100が必要
    
    st.info(f"確信度 **{min_confidence * 100}%** 以上のコメント **{len(df_filtered)}** 件で分析を表示しています。")

    sentiment_counts = df_filtered["感情"].value_counts()
    st.metric("ポジティブ", sentiment_counts.get("POSITIVE", 0))
    st.metric("ネガティブ", sentiment_counts.get("NEGATIVE", 0))
    st.metric("ニュートラル", sentiment_counts.get("NEUTRAL", 0))

        # --- 確信度分布グラフの追加 ---
    st.subheader("📊 モデルの確信度分布")
    if not df.empty:
        confidence_hist_base64 = plot_confidence_histogram(df)
        if confidence_hist_base64:
            st.image(confidence_hist_base64, caption="モデルが感情を判断した際の確信度スコアの分布")
        else:
            st.info("🤔 コメント数が少ないため、確信度分布グラフは生成できませんでした。")
    else:
        st.info("分析結果がないため、確信度分布グラフは表示できません。")

    st.subheader("感情の割合（円グラフ）")
    if not df.empty:
        pie_chart_base64 = plot_sentiment_pie(df_filtered.to_dict('records'))
        if pie_chart_base64: st.image(pie_chart_base64, caption="感情分析結果の円グラフ")
        else: st.info("🤔 コメント数が少ないため、感情の割合グラフは表示できないみたい...")
    else: st.info("分析データがないため、グラフは表示できません。")
    
    # コメントリストを一度だけ取得しておく
    positive_comments_list = df_filtered[df_filtered["感情"] == "POSITIVE"]["コメント"].tolist()
    negative_comments_list = df_filtered[df_filtered["感情"] == "NEGATIVE"]["コメント"].tolist()



        # ワードクラウド
    st.subheader("キーワード分析（ワードクラウド）")
    if positive_comments_list:
        st.write("#### ✨ ポジティブコメントのキーワード")
        wc_positive_base64 = create_wordcloud(positive_comments_list, background_color="white")
        if wc_positive_base64: st.image(wc_positive_base64)
        else: st.info("✍️ 十分なポジティブコメントがないため、ワードクラウドは生成できません。")
    
    if negative_comments_list:
        st.write("#### 😥 ネガティブコメントのキーワード")
        wc_negative_base64 = create_wordcloud(negative_comments_list, background_color="black")
        if wc_negative_base64: st.image(wc_negative_base64)
        else: st.info("✍️ 十分なネガティブコメントがないため、ワードクラウドは生成できません。")

        # 頻出単語の棒グラフ
    st.subheader("頻出単語のランキング")
    if positive_comments_list:
        st.write("#### ✨ ポジティブコメントでよく出るワード")
        positive_words_list = get_top_words(positive_comments_list)
        if positive_words_list:
            wc_positive_bar_base64 = plot_word_frequency_bar(positive_words_list, "ポジティブコメントの頻出単語", color='#FFB3C6')
            st.image(wc_positive_bar_base64)
            st.dataframe(pd.DataFrame(positive_words_list, columns=["単語", "件数"]), hide_index=True)
        else: st.info("ポジティブなコメントがないため、ランキングは表示できません。")
    
    if negative_comments_list:
        st.write("#### 😥 ネガティブコメントでよく出るワード")
        negative_words_list = get_top_words(negative_comments_list)
        if negative_words_list:
            wc_negative_bar_base64 = plot_word_frequency_bar(negative_words_list, "ネガティブコメントの頻出単語", color='#FF4B4B')
            st.image(wc_negative_bar_base64)
            st.dataframe(pd.DataFrame(negative_words_list, columns=["単語", "件数"]), hide_index=True)
        else: st.info("ネガティブなコメントがないため、ランキングは表示できません。")

# ... (代表コメントの表示セクション) ...

    # 代表コメントの表示
    st.subheader("代表的なコメント（ポジティブ）")
    for c in extract_top_comments(df_filtered, label="POSITIVE"):
        # --- ここを修正 ---
        # 頻出単語を取得して、それをハイライトする
        top_positive_words = get_top_words(df_filtered[df_filtered["感情"] == "POSITIVE"]["コメント"].tolist(), top_n=10)
        highlight_text = c
        for word, _ in top_positive_words:
            highlight_text = highlight_text.replace(word, f"**{word}**") # 太字にハイライト
        st.write(f"💬 {highlight_text}", unsafe_allow_html=True) # HTML表示を許可
    
    st.subheader("代表的なコメント（ネガティブ）")
    for c in extract_top_comments(df_filtered, label="NEGATIVE"):
        # --- ここも修正 ---
        top_negative_words = get_top_words(df_filtered[df_filtered["感情"] == "NEGATIVE"]["コメント"].tolist(), top_n=10)
        highlight_text = c
        for word, _ in top_negative_words:
            highlight_text = highlight_text.replace(word, f"**{word}**") # 太字にハイライト
        st.write(f"💬 {highlight_text}", unsafe_allow_html=True) # HTML表示を許可

    st.subheader("代表的なコメント（ニュートラル）")
    for c in extract_top_comments(df_filtered, label="NEUTRAL"):
        st.write(f"💬 {c}")

        # --- ここからクラスタリング機能のUIを追加 ---
    st.subheader("📊 コメントの自動クラスタリング")
    st.info("AIがコメントを話題ごとに自動で整理してくれるよ！")

        # クラスタリングモデルがロードされていればUIを表示
    if cluster_comments(comments=[], n_clusters=2) != "クラスタリングモデルのロードに失敗しました。": # 関数が動作するかチェック

    # クラスタ数をユーザーが指定できるようにする
        if not df.empty:
            n_clusters = st.slider(
                "🗂️ いくつの話題に分類する？",
                min_value=2,
                max_value=min(6, len(df) if len(df) >= 2 else 2), # コメント数が2件未満の場合は2が最小値
                value=min(3, len(df) if len(df) >= 2 else 2) # 初期値は5件 or コメント数
            )

            if st.button("✨ トピックを分析開始！", key="cluster_button"):
                
                

                    # --- ここで、状態に基づいてクラスタリングを実行し、結果を保存 ---
            
                with st.spinner(f"AIがコメントを {n_clusters} 個の話題に分類中..."):
                    
                    cluster_results, cluster_labels  = cluster_comments(df['コメント'].tolist(), n_clusters)

                    # 結果をセッションステートに保存
                    st.session_state.cluster_results = cluster_results

                    # 結果を df に追加してセッションステートに保存
                    df['cluster_label'] = cluster_labels
                    st.session_state.df = df
                    
            
                if isinstance(st.session_state.cluster_results, str):
                    st.warning(st.session_state.cluster_results) # コメントが少ない場合の警告
                    st.session_state.clustering_run = False # 失敗したらフラグを戻す
                else:
                        # クラスタリング結果をセッションステートに保存
                        df['cluster_label'] = cluster_labels
                        st.session_state.df = df # メインの df を更新
                        st.session_state.cluster_results = cluster_results
                        st.session_state.clustering_run = True # 成功したらフラグを立てる
                # --- セッションステートに結果があれば、常にクラスタリング結果を表示 ---
        if st.session_state.clustering_run and 'cluster_results' in st.session_state and isinstance(st.session_state.cluster_results, dict):
            st.write("#### 🔍 発見された話題グループ")
            for cluster_id, data in st.session_state.cluster_results.items():
                with st.expander(f"グループ {cluster_id + 1} ({data['comments_count']}件): 代表ワード - {', '.join(data['top_words'])}", expanded=True):
                    st.write(f"💬 **代表コメント:** {data['representative_comment']}")
                
                    # --- ここが修正 ---
                    # そのクラスターに属するコメントだけをフィルタリングして表示
                    cluster_comments_df = st.session_state.df[st.session_state.df['cluster_label'] == cluster_id]
                    st.dataframe(cluster_comments_df, hide_index=True)
                    
                    
                    # --- 修正ここまで ---
        else:
            st.info("クラスタリングを開始するには、ボタンを押してください。")
    else:
        st.info("クラスタリングを行うには、まず感情分析を実行してください。")
    # --- クラスタリング機能のUI追加ここまで ---


    st.subheader("💡 AIによる詳細分析")
    st.info("AIがコメントの背景にある理由や改善点を分析してくれます！")
    llm_keyword = st.text_input("🔍 **コメントの中から**、特に詳しく分析したいキーワードを入力してね (例: リップ, クッション, チーク)", key="llm_keyword_input")
    if llm_keyword:
        with st.expander(f"😥 「{llm_keyword}」に関するネガティブ分析結果", expanded=True):
            with st.spinner("AIがネガティブコメントから改善提案を考えているよ..."):
                llm_insight_neg = generate_insight_with_llm(df_filtered, "NEGATIVE", llm_keyword)
                st.markdown(llm_insight_neg)
        
        with st.expander(f"🎉 「{llm_keyword}」に関するポジティブ分析結果", expanded=True):
            with st.spinner("AIがポジティブコメントの強みを分析中だよ..."):
                llm_insight_pos = generate_insight_with_llm(df_filtered, "POSITIVE", llm_keyword)
                st.markdown(llm_insight_pos)

    # CSVダウンロード
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📝 分析結果をCSVでダウンロード", csv, "result.csv", "text/csv")


    




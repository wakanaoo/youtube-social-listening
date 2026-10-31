# services/youtube_collector.py

import streamlit as st
from youtubesearchpython import VideosSearch
import yt_dlp
from datetime import datetime, timedelta # timedelta もインポート

# 新しいヘルパー関数：相対的な時間表現をdatetimeオブジェクトに変換
def parse_relative_time(relative_time_str: str) -> datetime:
    """
    '2 years ago', '4 months ago', '1 day ago' などの相対的な時間表現を
    datetimeオブジェクトに変換します。
    変換できない場合は、デフォルトとして1970-01-01を返します。
    """
    now = datetime.now()
    parts = relative_time_str.lower().split()

    if len(parts) >= 2 and parts[0].isdigit():
        value = int(parts[0])
        unit = parts[1]

        if 'year' in unit:
            return now - timedelta(days=value * 365) # 簡易的に365日/年
        elif 'month' in unit:
            return now - timedelta(days=value * 30) # 簡易的に30日/月
        elif 'week' in unit:
            return now - timedelta(weeks=value)
        elif 'day' in unit:
            return now - timedelta(days=value)
        elif 'hour' in unit:
            return now - timedelta(hours=value)
        elif 'minute' in unit:
            return now - timedelta(minutes=value)
        elif 'second' in unit:
            return now - timedelta(seconds=value)
    
    # 'Just now' やその他の変換できないケース
    if 'just now' in relative_time_str.lower():
        return now
    
    return datetime(1970, 1, 1) # 変換できない場合のデフォルト

@st.cache_data(show_spinner=False)
def search_youtube_videos(keyword: str):
    """
    YouTube動画をキーワードで検索し、結果のリストを返す。
    公開日を含め、最新の動画が上に表示されるようにソートする。
    """
    videos_search = VideosSearch(keyword, limit=30)
    video_list_raw = videos_search.result()["result"]
    
    if not video_list_raw:
        print("DEBUG: video_list_raw is empty, returning []")
        return [
        f"{video['title']} - https://www.youtube.com/watch?v={video['id']}" 
        for video in video_list_raw
    ]

    sortable_videos = []
    for video in video_list_raw:
        # print(f"DEBUG: Processing video: {video.get('title', 'No Title')}")
        # print(f"DEBUG: Full video object: {video}") # デバッグ出力は不要になったら削除

        published_time_str = video.get('publishedTime')
        
        if published_time_str:
            published_dt = parse_relative_time(published_time_str)
        else:
            published_dt = datetime(1970, 1, 1)
            # print(f"DEBUG: publishedTime not found for {video.get('title', 'No Title')}, defaulting to 1970-01-01")

        sortable_videos.append((published_dt, video))

    # 最新の動画が上に来るように降順でソート
    sortable_videos.sort(key=lambda x: x[0], reverse=True)

    display_list = []
    for published_dt, video in sortable_videos:
        display_date = published_dt.strftime('%Y-%m-%d')
        display_list.append(f"{display_date} - {video['title']} - https://www.youtube.com/watch?v={video['id']}")
    
    return display_list

@st.cache_data(show_spinner=False)
def get_comments(selected_video_string: str) -> dict: # 戻り値の型をdictに変更
    """
    指定されたYouTube動画のコメント、タイトル、公開日を取得する。
    """
    try:
        # "タイトル - URL" の形式からURL部分を抽出
        parts = selected_video_string.split(' - ')
        if len(parts) < 2:
            st.error(f"選択された動画の形式が不正です: {selected_video_string}")
            return {'comments': [], 'title': '取得エラー', 'upload_date': '取得エラー'}

        video_url = parts[-1] # 最後の要素がURLであると想定

        # yt-dlp を使用して動画情報とコメントを取得
        ydl_opts = {
            'noplaylist': True,
            'extract_flat': 'comments',
            'getcomments': True,
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            comments = info_dict.get('comments', [])

         # コメントテキストのみを抽出
        comment_texts = [comment['text'] for comment in comments if 'text' in comment]

        # 動画のタイトルと公開日を取得
        video_title = info_dict.get('title', 'タイトル取得エラー')
        upload_date_str = info_dict.get('upload_date') # YYYYMMDD 形式で取得される
        
        # 公開日を YYYY-MM-DD 形式に変換
        if upload_date_str:
            # yt-dlpのupload_dateは YYYYMMDD 形式なので、パース
            upload_date = datetime.strptime(upload_date_str, '%Y%m%d').strftime('%Y-%m-%d')
        else:
            upload_date = '日付不明'

        # 結果を辞書形式で返す
        return {
            'comments': comment_texts,
            'title': video_title,
            'upload_date': upload_date
        }
        
    except Exception as e:
        # エラーメッセージに動画URLを含める
        st.error(f"コメントの取得に失敗しました。URL: {video_url}。エラー: {e}")
        return {'comments': [], 'title': '取得失敗', 'upload_date': '取得失敗'} # エラー時も辞書を返す
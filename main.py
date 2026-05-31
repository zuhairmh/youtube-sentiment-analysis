import os
import re
import io
import base64
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Ensure Matplotlib uses a non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Initialize NLTK/TextBlob Safely
import nltk
from textblob import TextBlob

try:
    # Safely download NLTK data if not present
    nltk.data.find('tokenizers/punkt')
except (LookupError, AttributeError):
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

app = Flask(__name__)

# Fallback basic rule-based sentiment analyzer in case TextBlob library fails
def analyze_sentiment(text):
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.05:
            return 'positive', polarity
        elif polarity < -0.05:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    except Exception as e:
        # Custom rule-based word-matching fallback for high reliability
        lower_text = text.lower()
        pos_words = {'good', 'great', 'awesome', 'excellent', 'happy', 'love', 'nice', 'best', 'wonderful', 'amazing', 'cool', 'perfect', 'helpful', 'beautiful', 'fun', 'classic'}
        neg_words = {'bad', 'worst', 'terrible', 'awful', 'sad', 'hate', 'worse', 'boring', 'useless', 'waste', 'disappointed', 'annoyed', 'broken', 'stupid', 'dumb', 'fail'}
        
        score = 0
        words = re.findall(r'\b\w+\b', lower_text)
        for w in words:
            if w in pos_words:
                score += 0.3
            elif w in neg_words:
                score -= 0.3
                
        if score > 0.1:
            return 'positive', min(1.0, score)
        elif score < -0.1:
            return 'negative', max(-1.0, score)
        else:
            return 'neutral', 0.0

def extract_video_id(url):
    url = url.strip()
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
        
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:m\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/live\/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)
            
    # Soft fallback parses
    if 'v=' in url:
        parts = url.split('v=')
        if len(parts) > 1:
            vid = parts[1][:11]
            if len(vid) == 11:
                return vid
    if 'shorts/' in url:
        parts = url.split('shorts/')
        if len(parts) > 1:
            vid = parts[1][:11]
            if len(vid) == 11:
                return vid
    if 'youtu.be/' in url:
        parts = url.split('youtu.be/')
        if len(parts) > 1:
            vid = parts[1][:11]
            if len(vid) == 11:
                return vid
                
    return None

def fetch_video_details(video_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    items = data.get('items', [])
    if not items:
        return None
        
    snippet = items[0].get('snippet', {})
    stats = items[0].get('statistics', {})
    
    return {
        'title': snippet.get('title', 'Unknown Title'),
        'channel': snippet.get('channelTitle', 'Unknown Channel'),
        'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url') or snippet.get('thumbnails', {}).get('default', {}).get('url'),
        'published_at': snippet.get('publishedAt', '')[:10],
        'views': int(stats.get('viewCount', 0)),
        'likes': int(stats.get('likeCount', 0))
    }

def generate_pie_chart(pos_pct, neu_pct, neg_pct):
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [pos_pct, neu_pct, neg_pct]
    colors = ['#10B981', '#6B7280', '#F43F5E'] # Emerald, Cool Gray, Rose
    explode = (0.05, 0, 0) if pos_pct > 0 else (0, 0, 0)
    
    # Configure styling for transparent backdrop and clean font weights
    plt.rcParams['text.color'] = '#E5E7EB'
    plt.rcParams['axes.labelcolor'] = '#E5E7EB'
    
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='none')
    
    if sum(sizes) == 0:
        sizes = [33.3, 33.3, 33.4]
        labels = ['No Data', 'No Data', 'No Data']
        colors = ['#374151', '#4B5563', '#1F2937']
        explode = (0, 0, 0)
        
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%',
        shadow=True, 
        startangle=140,
        textprops={'fontsize': 10, 'weight': 'bold'},
        wedgeprops={'edgecolor': '#1F2937', 'linewidth': 1.5, 'antialiased': True}
    )
    
    for text in texts:
        text.set_color('#E5E7EB')
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color('#FFFFFF')
        autotext.set_fontsize(11)
        
    ax.axis('equal')
    
    # Export layout
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=180)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json or {}
    url = data.get('url', '')
    limit = int(data.get('limit', 100))
    user_api_key = data.get('api_key', '').strip()
    
    # API key resolution: User Input -> Env Variable
    api_key = user_api_key or os.getenv('YOUTUBE_API_KEY')
    
    if not api_key:
        return jsonify({
            'success': False,
            'error': 'YouTube API Key is missing. Please provide it in the input panel or configure it on the server dotenv environment.'
        }), 400
        
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({
            'success': False,
            'error': 'Invalid YouTube URL. Please verify the URL format and try again.'
        }), 400
        
    try:
        # 1. Fetch Video Metadata
        video_details = fetch_video_details(video_id, api_key)
        if not video_details:
            # If API key is completely wrong or video private
            return jsonify({
                'success': False,
                'error': 'Could not fetch video info. Please double-check your API Key and check if the YouTube URL is public.'
            }), 400
            
        # 2. Fetch Comments
        comments = []
        next_page_token = None
        comments_to_fetch = min(limit, 500) # Safe capping
        
        while len(comments) < comments_to_fetch:
            page_size = min(100, comments_to_fetch - len(comments))
            api_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}&maxResults={page_size}&textFormat=plainText"
            if next_page_token:
                api_url += f"&pageToken={next_page_token}"
                
            res = requests.get(api_url)
            if res.status_code != 200:
                err_msg = 'Failed to fetch comments from YouTube API.'
                try:
                    err_json = res.json()
                    err_msg = err_json.get('error', {}).get('message', err_msg)
                except Exception:
                    pass
                raise Exception(err_msg)
                
            res_data = res.json()
            items = res_data.get('items', [])
            if not items:
                break
                
            for item in items:
                snippet = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                text = snippet.get('textDisplay', '')
                author = snippet.get('authorDisplayName', 'Anonymous')
                avatar = snippet.get('authorProfileImageUrl', '')
                like_count = snippet.get('likeCount', 0)
                published_at = snippet.get('publishedAt', '')
                
                if text:
                    comments.append({
                        'text': text,
                        'author': author,
                        'avatar': avatar,
                        'likes': int(like_count),
                        'date': published_at[:10]
                    })
                    
            next_page_token = res_data.get('nextPageToken')
            if not next_page_token:
                break
                
        if not comments:
            return jsonify({
                'success': False,
                'error': 'No public comments found on this video.'
            }), 400
            
        # 3. Analyze Comments Sentiment
        pos_list = []
        neu_list = []
        neg_list = []
        
        total_polarity = 0
        
        for c in comments:
            label, score = analyze_sentiment(c['text'])
            c['sentiment'] = label
            c['score'] = round(score, 2)
            total_polarity += score
            
            if label == 'positive':
                pos_list.append(c)
            elif label == 'negative':
                neg_list.append(c)
            else:
                neu_list.append(c)
                
        total_count = len(comments)
        pos_count = len(pos_list)
        neu_count = len(neu_list)
        neg_count = len(neg_list)
        
        pos_pct = round((pos_count / total_count) * 100, 1)
        neu_pct = round((neu_count / total_count) * 100, 1)
        neg_pct = round((neg_count / total_count) * 100, 1)
        
        avg_score = round(total_polarity / total_count, 2)
        
        # Formulate general mood
        if avg_score > 0.35:
            mood = 'Overwhelmingly Positive'
            mood_class = 'mood-over-positive'
        elif avg_score > 0.05:
            mood = 'Generally Positive'
            mood_class = 'mood-positive'
        elif avg_score > -0.05:
            mood = 'Mixed or Neutral'
            mood_class = 'mood-neutral'
        elif avg_score > -0.35:
            mood = 'Generally Negative'
            mood_class = 'mood-negative'
        else:
            mood = 'Overwhelmingly Negative'
            mood_class = 'mood-over-negative'
            
        # 4. Generate Matplotlib Base64 Image
        chart_base64 = generate_pie_chart(pos_pct, neu_pct, neg_pct)
        
        # Limit sample comments size for the JSON response
        sample_size = 20
        response_data = {
            'success': True,
            'video_details': video_details,
            'stats': {
                'total_analyzed': total_count,
                'positive_count': pos_count,
                'neutral_count': neu_count,
                'negative_count': neg_count,
                'positive_pct': pos_pct,
                'neutral_pct': neu_pct,
                'negative_pct': neg_pct,
                'average_score': avg_score,
                'mood': mood,
                'mood_class': mood_class
            },
            'chart_image': chart_base64,
            'samples': {
                'positive': pos_list[:sample_size],
                'neutral': neu_list[:sample_size],
                'negative': neg_list[:sample_size]
            },
            # Return all clean texts to build a dynamic word frequency client-side
            'all_comments_text': [c['text'] for c in comments]
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os
from matplotlib.font_manager import FontProperties

# --- 全新、更可靠的中文配置 ---
def get_chinese_font():
    """查找一个可用的中文字体文件路径"""
    possible_font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', # wqy-zenhei
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', # Noto Sans CJK
        'C:/Windows/Fonts/simhei.ttf', # Windows 黑体
        '/System/Library/Fonts/STHeiti Medium.ttc' # macOS 黑体
    ]
    for p in possible_font_paths:
        if os.path.exists(p):
            print(f"找到并使用中文字体: {p}")
            return FontProperties(fname=p)
    print("警告：未找到指定的中文字体文件。静态图中的中文可能无法正常显示。")
    return None

# 获取字体属性对象
CHINESE_FONT = get_chinese_font()

# 设置全局字体（如果找到了字体）
if CHINESE_FONT:
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT.get_name()]
    plt.rcParams['axes.unicode_minus'] = False 
# ------------------------------------

# 设置 seaborn 样式
sns.set(style="whitegrid")

def plot_rating_distribution(df, output_dir='outputs/figures'):
    """
    绘制评分分布图
    """
    os.makedirs(output_dir, exist_ok=True)
    rating_counts = df['rating'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    # 使用字体属性对象设置中文
    plt.title('评分分布', fontproperties=CHINESE_FONT)
    plt.xlabel('评分', fontproperties=CHINESE_FONT)
    plt.ylabel('数量', fontproperties=CHINESE_FONT)
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = px.bar(x=rating_counts.index, y=rating_counts.values, labels={'x': '评分', 'y': '数量'}, title='评分分布')
    return fig

def plot_aspect_scores(df, output_dir='outputs/figures'):
    """
    绘制各方面评分的雷达图和条形图
    """
    os.makedirs(output_dir, exist_ok=True)
    aspect_cols = [col for col in df.columns if col.endswith('_score')]
    if not aspect_cols: return None

    aspect_means = {col.replace('_score', ''): df[col].mean() for col in aspect_cols}
    aspect_means = {k: v for k, v in sorted(aspect_means.items(), key=lambda item: item[1], reverse=True)}
    
    aspect_translation = {'sound_quality': '音质', 'comfort': '舒适度', 'battery': '电池', 'connectivity': '连接性', 'noise_cancellation': '降噪', 'build_quality': '做工', 'controls': '控制', 'price': '价格', 'microphone': '麦克风', 'design': '设计'}
    
    categories = [aspect_translation.get(k, k) for k in aspect_means.keys()]
    values = [v if pd.notna(v) else 0 for v in aspect_means.values()]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='平均分数'))
    fig.update_layout(title_text="音频设备各方面评分")

    plt.figure(figsize=(12, 8))
    bars = plt.barh(categories, values)
    plt.title('各方面平均评分', fontproperties=CHINESE_FONT)
    plt.xlabel('平均分数 (1-10)', fontproperties=CHINESE_FONT)
    plt.yticks(fontproperties=CHINESE_FONT) # y轴刻度也需要设置
    plt.xlim(0, 10)
    for i, v in enumerate(values):
        plt.text(v + 0.1, i, f"{v:.2f}", va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_scores_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def plot_price_range_comparison(df, output_dir='outputs/figures'):
    """
    比较不同价格区间产品的方面评分
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'price_range' not in df.columns: return None
    
    aspect_cols = [col for col in df.columns if col.endswith('_score')]
    if not aspect_cols: return None
    
    aspect_translation = {'sound_quality': '音质', 'comfort': '舒适度', 'battery': '电池', 'connectivity': '连接性', 'noise_cancellation': '降噪', 'build_quality': '做工', 'controls': '控制', 'price': '价格', 'microphone': '麦克风', 'design': '设计'}
    
    price_ranges = df['price_range'].unique()
    fig = go.Figure()

    for price_range in sorted(price_ranges):
        price_df = df[df['price_range'] == price_range]
        if len(price_df) < 10: continue
            
        aspect_means = {col.replace('_score', ''): price_df[col].mean() for col in aspect_cols}
        aspect_means = {k: v for k, v in sorted(aspect_means.items())}
        
        fig.add_trace(go.Scatter(
            x=[aspect_translation.get(k, k) for k in aspect_means.keys()],
            y=[v if pd.notna(v) else 0 for v in aspect_means.values()],
            mode='lines+markers', name=price_range
        ))
    
    fig.update_layout(title_text="不同价格区间的方面评分", legend_title_text="价格区间")
    return fig

def plot_brand_comparison(df, output_dir='outputs/figures', min_reviews=30):
    """
    比较不同品牌产品的方面评分
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'brand' not in df.columns: return None
    
    aspect_cols = [col for col in df.columns if col.endswith('_score')]
    if not aspect_cols: return None

    aspect_translation = {'sound_quality': '音质', 'comfort': '舒适度', 'battery': '电池', 'connectivity': '连接性', 'noise_cancellation': '降噪', 'build_quality': '做工', 'controls': '控制', 'price': '价格', 'microphone': '麦克风', 'design': '设计'}
    
    brand_counts = df['brand'].value_counts()
    valid_brands = brand_counts[brand_counts >= min_reviews].index.tolist()[:10]
    if not valid_brands: return None
        
    brand_data = [{'Brand': brand, 'Aspect': aspect_translation.get(col.replace('_score', ''), col.replace('_score', '')), 'Score': df[df['brand'] == brand][col].mean()}
                  for brand in valid_brands for col in aspect_cols]
    
    brand_scores_df = pd.DataFrame(brand_data)
    if brand_scores_df.empty: return None

    heatmap_data = brand_scores_df.pivot(index='Brand', columns='Aspect', values='Score').fillna(0)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=1, vmax=10, fmt='.2f')
    plt.title('品牌方面评分比较', fontproperties=CHINESE_FONT)
    plt.xlabel('方面', fontproperties=CHINESE_FONT)
    plt.ylabel('品牌', fontproperties=CHINESE_FONT)
    plt.xticks(rotation=45, fontproperties=CHINESE_FONT)
    plt.yticks(fontproperties=CHINESE_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brand_comparison_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = px.imshow(heatmap_data, labels=dict(x="方面", y="品牌", color="分数"), title='品牌方面评分比较')
    return fig

def plot_word_clouds(df, output_dir='outputs/figures'):
    """
    为高评分和低评分评论创建词云
    """
    os.makedirs(output_dir, exist_ok=True)
    high_rated = df[df['rating'] >= 4]['clean_review_text'].dropna()
    low_rated = df[df['rating'] <= 2]['clean_review_text'].dropna()
    if high_rated.empty or low_rated.empty: return

    high_text, low_text = ' '.join(high_rated), ' '.join(low_rated)
    
    wordcloud_high = WordCloud(font_path=CHINESE_FONT.get_file() if CHINESE_FONT else None, width=800, height=400, background_color='white', max_words=200, colormap='viridis').generate(high_text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud_high, interpolation='bilinear')
    plt.axis('off'); plt.title('高评分评论词云 (4-5星)', fontproperties=CHINESE_FONT); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'high_rated_wordcloud.png'), dpi=300, bbox_inches='tight'); plt.close()
    
    wordcloud_low = WordCloud(font_path=CHINESE_FONT.get_file() if CHINESE_FONT else None, width=800, height=400, background_color='white', max_words=200, colormap='inferno').generate(low_text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud_low, interpolation='bilinear')
    plt.axis('off'); plt.title('低评分评论词云 (1-2星)', fontproperties=CHINESE_FONT); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low_rated_wordcloud.png'), dpi=300, bbox_inches='tight'); plt.close()

def create_all_visualizations(df, output_dir='outputs/figures'):
    """
    创建所有可视化
    """
    plot_rating_distribution(df, output_dir)
    plot_aspect_scores(df, output_dir)
    plot_price_range_comparison(df, output_dir)
    plot_brand_comparison(df, output_dir)
    plot_word_clouds(df, output_dir)
    print("所有可视化已创建完成。")
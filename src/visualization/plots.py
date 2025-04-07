import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os

# 设置绘图样式
plt.style.use('ggplot')
sns.set(style="whitegrid")

def plot_rating_distribution(df, output_dir='outputs/figures'):
    """
    绘制评分分布图
    
    Args:
        df (pd.DataFrame): 评论数据
        output_dir (str): 输出目录
    
    Returns:
        None
    """
    print("绘制评分分布...")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算评分分布
    rating_counts = df['rating'].value_counts().sort_index()
    
    # 创建绘图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title('评分分布')
    plt.xlabel('评分')
    plt.ylabel('数量')
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用Plotly创建交互式图表
    fig = px.bar(
        x=rating_counts.index, 
        y=rating_counts.values,
        labels={'x': '评分', 'y': '数量'},
        title='评分分布',
        color=rating_counts.index,
        color_continuous_scale='Viridis'
    )
    
    fig.write_html(os.path.join(output_dir, 'rating_distribution.html'))
    
    print(f"评分分布图已保存至 {output_dir}")

def plot_aspect_scores(df, output_dir='outputs/figures'):
    """
    绘制各方面评分的雷达图
    
    Args:
        df (pd.DataFrame): 包含方面评分的数据
        output_dir (str): 输出目录
    
    Returns:
        None
    """
    print("绘制方面评分...")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 识别方面评分列
    aspect_cols = [col for col in df.columns if col.endswith('_score')]
    if not aspect_cols:
        print("未找到方面评分列。")
        return
    
    # 计算每个方面的平均得分
    aspect_means = {}
    for col in aspect_cols:
        aspect = col.replace('_score', '')
        aspect_means[aspect] = df[col].mean()
    
    # 排序并整理数据
    aspect_means = {k: v for k, v in sorted(aspect_means.items(), key=lambda item: item[1], reverse=True)}
    
    # 翻译方面名称为中文
    aspect_translation = {
        'sound_quality': '音质',
        'comfort': '舒适度',
        'battery': '电池',
        'connectivity': '连接性',
        'noise_cancellation': '降噪',
        'build_quality': '做工',
        'controls': '控制',
        'price': '价格',
        'microphone': '麦克风',
        'design': '设计'
    }
    
    # 准备雷达图数据
    categories = [aspect_translation.get(k, k) for k in aspect_means.keys()]
    values = list(aspect_means.values())
    
    # 确保没有NaN值
    values = [0 if np.isnan(v) else v for v in values]
    
    # 创建雷达图
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='平均分数'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="音频设备各方面评分"
    )
    
    fig.write_html(os.path.join(output_dir, 'aspect_scores_radar.html'))
    
    # 创建条形图
    plt.figure(figsize=(12, 8))
    bars = plt.barh([aspect_translation.get(k, k) for k in aspect_means.keys()], list(aspect_means.values()))
    
    # 设置颜色渐变
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i / len(bars)))
    
    plt.title('各方面平均评分')
    plt.xlabel('平均分数 (1-10)')
    plt.xlim(0, 10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(aspect_means.values()):
        plt.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_scores_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"方面评分图已保存至 {output_dir}")

def plot_price_range_comparison(df, output_dir='outputs/figures'):
    """
    比较不同价格区间产品的方面评分
    
    Args:
        df (pd.DataFrame): 包含方面评分和价格区间的数据
        output_dir (str): 输出目录
    
    Returns:
        None
    """
    print("绘制价格区间比较...")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查必要的列是否存在
    if 'price_range' not in df.columns:
        print("未找到price_range列。")
        return
    
    # 识别方面评分列
    aspect_cols = [col for col in df.columns if col.endswith('_score')]
    if not aspect_cols:
        print("未找到方面评分列。")
        return
    
    # 翻译方面名称为中文
    aspect_translation = {
        'sound_quality': '音质',
        'comfort': '舒适度',
        'battery': '电池',
        'connectivity': '连接性',
        'noise_cancellation': '降噪',
        'build_quality': '做工',
        'controls': '控制',
        'price': '价格',
        'microphone': '麦克风',
        'design': '设计'
    }
    
    # 为每个价格区间计算平均方面评分
    price_ranges = df['price_range'].unique()
    
    # 创建多子图
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['不同价格区间的方面评分']
    )
    
    # 为每个价格区间添加一条折线
    for price_range in price_ranges:
        price_df = df[df['price_range'] == price_range]
        
        if len(price_df) < 10:  # 跳过样本太少的价格区间
            continue
            
        # 计算这个价格区间的方面平均分
        aspect_means = {}
        for col in aspect_cols:
            aspect = col.replace('_score', '')
            aspect_means[aspect] = price_df[col].mean()
        
        # 排序
        aspect_means = {k: v for k, v in sorted(aspect_means.items(), key=lambda item: item[0])}
        
        # 添加折线
        fig.add_trace(
            go.Scatter(
                x=[aspect_translation.get(k, k) for k in aspect_means.keys()],
                y=list(aspect_means.values()),
                mode='lines+markers',
                name=price_range
            ),
            row=1, col=1
        )
    
    # 更新布局
    fig.update_layout(
        title="不同价格区间的方面评分",
        xaxis_title="方面",
        yaxis_title="平均分数 (1-10)",
        legend_title="价格区间",
        yaxis=dict(range=[0, 10])
    )
    
    fig.write_html(os.path.join(output_dir, 'price_range_comparison.html'))
    
    print(f"价格区间比较图已保存至 {output_dir}")

def plot_brand_comparison(df, output_dir='outputs/figures', min_reviews=30):
    """
    比较不同品牌产品的方面评分
    
    Args:
        df (pd.DataFrame): 包含方面评分和品牌的数据
        output_dir (str): 输出目录
        min_reviews (int): 最少评论数量的阈值
    
    Returns:
        None
    """
    print("绘制品牌比较...")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查必要的列是否存在
    if 'brand' not in df.columns:
        print("未找到brand列。")
        return
    
    # 识别方面评分列
    aspect_cols = [col for col in df.columns if col.endswith('_score')]
    if not aspect_cols:
        print("未找到方面评分列。")
        return
    
    # 翻译方面名称为中文
    aspect_translation = {
        'sound_quality': '音质',
        'comfort': '舒适度',
        'battery': '电池',
        'connectivity': '连接性',
        'noise_cancellation': '降噪',
        'build_quality': '做工',
        'controls': '控制',
        'price': '价格',
        'microphone': '麦克风',
        'design': '设计'
    }
    
    # 计算每个品牌的评论数
    brand_counts = df['brand'].value_counts()
    
    # 只保留有足够评论的品牌
    valid_brands = brand_counts[brand_counts >= min_reviews].index.tolist()
    
    if not valid_brands:
        print(f"没有品牌的评论数量至少为 {min_reviews}。")
        return
    
    # 将品牌限制在前10个
    valid_brands = valid_brands[:10]
    
    # 为每个品牌计算平均方面评分
    brand_data = []
    
    for brand in valid_brands:
        brand_df = df[df['brand'] == brand]
        
        # 计算这个品牌的方面平均分
        aspect_means = {}
        for col in aspect_cols:
            aspect = col.replace('_score', '')
            aspect_means[aspect] = brand_df[col].mean()
        
        # 添加到列表
        for aspect, score in aspect_means.items():
            brand_data.append({
                'Brand': brand,
                'Aspect': aspect_translation.get(aspect, aspect),
                'Score': score
            })
    
    # 创建数据框
    brand_scores_df = pd.DataFrame(brand_data)
    
    # 创建热力图
    plt.figure(figsize=(14, 10))
    heatmap_data = brand_scores_df.pivot(index='Brand', columns='Aspect', values='Score')
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=1, vmax=10, fmt='.2f')
    plt.title('品牌方面评分比较')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brand_comparison_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建交互式热力图
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="方面", y="品牌", color="分数"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        aspect="auto",
        color_continuous_scale='Viridis',
        range_color=[1, 10],
        title='品牌方面评分比较'
    )
    
    fig.update_layout(
        xaxis_title="方面",
        yaxis_title="品牌"
    )
    
    fig.write_html(os.path.join(output_dir, 'brand_comparison_heatmap.html'))
    
    print(f"品牌比较图已保存至 {output_dir}")

def plot_word_clouds(df, output_dir='outputs/figures'):
    """
    为高评分和低评分评论创建词云
    
    Args:
        df (pd.DataFrame): 评论数据
        output_dir (str): 输出目录
    
    Returns:
        None
    """
    print("创建词云...")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 分离高评分和低评分评论
    high_rated = df[df['rating'] >= 4]['clean_review_text'].dropna()
    low_rated = df[df['rating'] <= 2]['clean_review_text'].dropna()
    
    if len(high_rated) == 0 or len(low_rated) == 0:
        print("数据不足以创建词云。")
        return
    
    # 合并文本
    high_text = ' '.join(high_rated)
    low_text = ' '.join(low_rated)
    
    # 创建高评分词云
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=200,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate(high_text)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('高评分评论词云 (4-5星)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'high_rated_wordcloud.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建低评分词云
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=200,
        colormap='inferno',
        contour_width=1,
        contour_color='firebrick'
    ).generate(low_text)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('低评分评论词云 (1-2星)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low_rated_wordcloud.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"词云已保存至 {output_dir}")

def create_all_visualizations(df, output_dir='outputs/figures'):
    """
    创建所有可视化
    
    Args:
        df (pd.DataFrame): 带有方面分析的评论数据
        output_dir (str): 输出目录
    
    Returns:
        None
    """
    plot_rating_distribution(df, output_dir)
    plot_aspect_scores(df, output_dir)
    plot_price_range_comparison(df, output_dir)
    plot_brand_comparison(df, output_dir)
    plot_word_clouds(df, output_dir)
    
    print("所有可视化已创建完成。")

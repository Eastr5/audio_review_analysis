import pandas as pd
import numpy as np
import spacy
from collections import defaultdict
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# 确保下载必要的NLTK资源
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# 加载spaCy模型
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("正在下载spaCy模型...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# 初始化情感分析器
sid = SentimentIntensityAnalyzer()

# 定义音频设备相关的方面及关键词
AUDIO_ASPECTS = {
    'sound_quality': ['sound', 'audio', 'quality', 'clarity', 'bass', 'treble', 'mid', 'tone', 'frequency'],
    'comfort': ['comfort', 'comfortable', 'fit', 'ear', 'cushion', 'padding', 'weight', 'light', 'heavy'],
    'battery': ['battery', 'charge', 'life', 'duration', 'hour', 'lasting', 'power'],
    'connectivity': ['bluetooth', 'connection', 'pair', 'wireless', 'range', 'distance', 'stable'],
    'noise_cancellation': ['noise', 'cancellation', 'anc', 'isolation', 'ambient', 'background'],
    'build_quality': ['build', 'quality', 'material', 'durable', 'sturdy', 'plastic', 'metal', 'robust'],
    'controls': ['control', 'button', 'touch', 'volume', 'skip', 'pause', 'play', 'responsive'],
    'price': ['price', 'cost', 'value', 'worth', 'expensive', 'cheap', 'affordable'],
    'microphone': ['mic', 'microphone', 'call', 'voice', 'speak', 'talking'],
    'design': ['design', 'look', 'style', 'color', 'appearance', 'aesthetic']
}

def extract_aspect_sentences(text, aspects_dict=AUDIO_ASPECTS):
    """
    从评论文本中提取与各方面相关的句子
    
    Args:
        text (str): 评论文本
        aspects_dict (dict): 方面及对应关键词字典
    
    Returns:
        dict: 各方面及对应的句子
    """
    if not isinstance(text, str) or not text:
        return {}
    
    # 分句
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 存储各方面的句子
    aspect_sentences = defaultdict(list)
    
    # 遍历句子，匹配方面
    for sentence in sentences:
        sentence = sentence.lower()
        for aspect, keywords in aspects_dict.items():
            if any(keyword in sentence for keyword in keywords):
                aspect_sentences[aspect].append(sentence)
    
    return dict(aspect_sentences)

def analyze_aspect_sentiment(aspect_sentences):
    """
    分析各方面的情感得分
    
    Args:
        aspect_sentences (dict): 各方面及对应的句子
    
    Returns:
        dict: 各方面的情感得分
    """
    aspect_sentiments = {}
    
    for aspect, sentences in aspect_sentences.items():
        if not sentences:
            continue
        
        # 计算每个句子的情感得分
        sentiment_scores = [sid.polarity_scores(sentence)['compound'] for sentence in sentences]
        
        # 计算平均情感得分
        aspect_sentiments[aspect] = {
            'score': np.mean(sentiment_scores),
            'sentence_count': len(sentences),
            'sentences': sentences,
            'positive_example': sentences[np.argmax(sentiment_scores)] if sentiment_scores else "",
            'negative_example': sentences[np.argmin(sentiment_scores)] if sentiment_scores else ""
        }
    
    return aspect_sentiments

def extract_aspects_batch(reviews_df, text_column='clean_review_text', max_samples=None):
    """
    批量提取评论中的方面及情感
    
    Args:
        reviews_df (pd.DataFrame): 评论数据
        text_column (str): 文本列名
        max_samples (int, optional): 最大处理样本数
    
    Returns:
        pd.DataFrame: 包含方面分析结果的数据
    """
    print("正在从评论中提取方面...")
    result_df = reviews_df.copy()
    
    # 限制处理样本数
    if max_samples and len(result_df) > max_samples:
        result_df = result_df.sample(max_samples, random_state=42)
    
    # 创建用于存储结果的列
    result_df['aspect_sentences'] = None
    result_df['aspect_sentiments'] = None
    
    # 逐条处理评论
    aspect_sentences_list = []
    aspect_sentiments_list = []
    
    for idx, row in result_df.iterrows():
        text = row[text_column]
        
        # 提取方面句子
        aspect_sentences = extract_aspect_sentences(text)
        aspect_sentences_list.append(aspect_sentences)
        
        # 分析方面情感
        aspect_sentiments = analyze_aspect_sentiment(aspect_sentences)
        aspect_sentiments_list.append(aspect_sentiments)
    
    # 添加到DataFrame
    result_df['aspect_sentences'] = aspect_sentences_list
    result_df['aspect_sentiments'] = aspect_sentiments_list
    
    print("方面提取完成。")
    return result_df

def generate_aspect_scores(reviews_with_aspects):
    """
    生成各方面的评分
    
    Args:
        reviews_with_aspects (pd.DataFrame): 带有方面分析的评论数据
    
    Returns:
        pd.DataFrame: 方面评分数据
    """
    print("正在生成方面评分...")
    # 获取所有方面
    all_aspects = list(AUDIO_ASPECTS.keys())
    
    # 创建用于存储结果的DataFrame
    result_df = reviews_with_aspects.copy()
    
    # 为每个方面创建得分列
    for aspect in all_aspects:
        result_df[f'{aspect}_score'] = None
        result_df[f'{aspect}_count'] = 0
    
    # 计算方面得分
    for idx, row in result_df.iterrows():
        aspect_sentiments = row['aspect_sentiments']
        if not aspect_sentiments:
            continue
            
        for aspect, data in aspect_sentiments.items():
            result_df.at[idx, f'{aspect}_score'] = data['score']
            result_df.at[idx, f'{aspect}_count'] = data['sentence_count']
    
    # 将得分规范化到1-10范围
    for aspect in all_aspects:
        score_col = f'{aspect}_score'
        # 将-1到1的得分映射到1-10
        mask = result_df[score_col].notna()
        result_df.loc[mask, score_col] = ((result_df.loc[mask, score_col] + 1) / 2) * 9 + 1
    
    print("方面评分完成。")
    return result_df

def create_review_aspects_dataset(df, output_path='data/processed/review_aspects.csv'):
    """
    创建带有方面分析的完整数据集
    
    Args:
        df (pd.DataFrame): 预处理后的评论数据
        output_path (str): 输出路径
    
    Returns:
        pd.DataFrame: 包含方面分析的数据集
    """
    # 检查是否已存在处理好的数据
    try:
        result_df = pd.read_csv(output_path)
        print(f"从 {output_path} 加载方面分析结果")
        return result_df
    except FileNotFoundError:
        pass
    
    # 提取方面信息
    df_with_aspects = extract_aspects_batch(df)
    
    # 生成方面评分
    result_df = generate_aspect_scores(df_with_aspects)
    
    # 保存结果
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将字典列转换为字符串以便保存
    result_df['aspect_sentences'] = result_df['aspect_sentences'].apply(lambda x: str(x) if x else None)
    result_df['aspect_sentiments'] = result_df['aspect_sentiments'].apply(lambda x: str(x) if x else None)
    
    # 保存
    result_df.to_csv(output_path, index=False)
    print(f"已将方面分析结果保存至 {output_path}")
    
    return result_df

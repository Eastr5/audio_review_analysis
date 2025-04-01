# 音频设备评论分析系统详细文档

## 📋 项目概述

音频设备评论分析系统是一个基于Python的数据分析工具，专门用于从Amazon电子产品评论中提取、分析和可视化与音频设备相关的用户评价。该系统能够自动识别与音频设备相关的评论，并从评论文本中提取关于音质、舒适度、电池寿命等10个关键方面的评分和情感倾向，最终通过多种可视化图表直观展示分析结果。

### 🔍 核心功能

- **智能评论筛选**：从电子产品评论数据集中自动筛选出音频设备相关评论
- **多维度评分提取**：基于NLP技术从评论文本中提取10个关键方面的评分
- **品牌与价格区间比较**：对不同品牌和价格区间产品在各方面的表现进行比较分析
- **情感分析**：分析用户对各个方面的情感倾向
- **直观数据可视化**：通过雷达图、热力图、条形图等多种方式展示分析结果
- **PowerBI导出支持**：支持导出处理后的数据用于PowerBI创建高级可视化和报表

## 🏗️ 项目架构

项目采用模块化设计，主要包含以下组件：

```
audio_review_analysis/
├── data/                      # 数据存储
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后的数据
├── outputs/                   # 输出文件
│   └── figures/               # 生成的可视化
├── src/                       # 源代码
│   ├── data/                  # 数据获取与预处理
│   │   ├── __init__.py
│   │   ├── acquisition.py     # 数据获取模块
│   │   └── preprocessing.py   # 数据预处理模块
│   ├── features/              # 特征工程
│   │   ├── __init__.py
│   │   └── aspect_extraction.py # 方面提取与分析
│   └── visualization/         # 可视化模块
│       ├── __init__.py
│       └── plots.py           # 绘图功能
├── config.py                  # 全局配置文件
├── main.py                    # 主程序入口
└── requirements.txt           # 项目依赖
```

### 📊 数据流程

1. **数据获取**：从Amazon下载电子产品评论数据
2. **数据预处理**：清洗文本、分词、提取基本特征
3. **方面提取**：识别评论中与各个方面相关的句子并进行情感分析
4. **评分生成**：为每个方面生成标准化评分
5. **数据可视化**：创建多种图表展示分析结果

## 🛠️ 安装与设置

### 环境要求

- Python 3.6+
- pip 包管理器

### 安装步骤

1. **克隆仓库（或创建项目目录）**

```bash
mkdir audio_review_analysis
cd audio_review_analysis
```

2. **创建目录结构**

```bash
mkdir -p data/raw data/processed outputs/figures
mkdir -p src/data src/features src/visualization
```

3. **安装依赖**

```bash
pip install pandas numpy matplotlib seaborn plotly nltk spacy scikit-learn wordcloud tqdm requests openpyxl
python -m nltk.downloader punkt vader_lexicon stopwords
python -m spacy download en_core_web_sm
```

## 📝 详细组件说明

# 完整音频设备评论分析系统代码

以下是项目的完整代码，包括所有模块和新添加的Streamlit仪表盘。

## config.py

```python
"""
配置文件，包含项目的全局配置
"""

# 数据相关配置
DATA_CONFIG = {
    'amazon_url': "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'sample_size': None  # 设置为整数以限制处理的样本数量，用于测试
}

# 音频设备关键词
AUDIO_KEYWORDS = [
    'headphone', 'earphone', 'earbud', 'headset', 
    'earpiece', 'airpod', 'speaker', 'soundbar',
    'audio', 'sound', 'bluetooth speaker', 'wireless headphone'
]

# 方面分析配置
ASPECT_CONFIG = {
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

# 可视化配置
VIZ_CONFIG = {
    'output_dir': 'outputs/figures',
    'color_scheme': 'viridis',
    'min_reviews_per_brand': 30,
    'min_reviews_per_price_range': 10
}
```

## src/data/acquisition.py

```python
import os
import gzip
import json
import requests
import pandas as pd
from tqdm import tqdm

def download_amazon_dataset(url, target_path):
    """
    下载Amazon评论数据集
    
    Args:
        url (str): 数据集URL
        target_path (str): 保存路径
    
    Returns:
        str: 数据集保存的本地路径
    """
    # 创建目录
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # 如果文件已存在，跳过下载
    if os.path.exists(target_path):
        print(f"文件已存在: {target_path}")
        return target_path
    
    # 下载数据
    print(f"正在从 {url} 下载数据...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm(
        desc=target_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    print(f"下载完成。文件已保存至 {target_path}")
    return target_path

def load_amazon_dataset(file_path):
    """
    加载并解析Amazon评论数据集
    
    Args:
        file_path (str): 数据集文件路径(.json.gz)
    
    Returns:
        pd.DataFrame: 包含评论数据的DataFrame
    """
    print(f"正在从 {file_path} 加载数据...")
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            data.append(json.loads(line.strip()))
            # 测试时可以限制加载数量
            # if i > 100000:  # 加载前10万条数据
            #     break
    
    df = pd.DataFrame(data)
    print(f"已加载 {len(df)} 条评论")
    return df

def filter_audio_products(df, keywords=None):
    """
    从电子产品评论中筛选音频设备相关评论
    
    Args:
        df (pd.DataFrame): 电子产品评论DataFrame
        keywords (list, optional): 音频设备关键词列表
    
    Returns:
        pd.DataFrame: 音频设备相关评论
    """
    if keywords is None:
        keywords = [
            'headphone', 'earphone', 'earbud', 'headset', 
            'earpiece', 'airpod', 'speaker', 'soundbar',
            'audio', 'sound', 'bluetooth speaker', 'wireless headphone'
        ]
    
    # 在产品名称或评论中匹配关键词
    keyword_pattern = '|'.join(keywords)
    
    # 检查评论标题
    title_mask = df['summary'].str.contains(
        keyword_pattern, case=False, na=False
    )
    
    # 检查评论内容
    text_mask = df['reviewText'].str.contains(
        keyword_pattern, case=False, na=False
    )
    
    # 合并筛选条件
    filtered_df = df[title_mask | text_mask].copy()
    
    print(f"找到 {len(filtered_df)} 条音频设备相关评论")
    return filtered_df

def get_audio_dataset(url=None, save_dir='data/raw', processed_dir='data/processed'):
    """
    获取并处理音频设备评论数据集
    
    Args:
        url (str, optional): 数据集URL
        save_dir (str): 原始数据保存目录
        processed_dir (str): 处理后数据保存目录
    
    Returns:
        pd.DataFrame: 处理后的音频设备评论数据
    """
    # 默认使用Amazon电子产品评论数据集
    if url is None:
        url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
    
    # 设置文件路径
    filename = url.split('/')[-1]
    raw_path = os.path.join(save_dir, filename)
    processed_path = os.path.join(processed_dir, 'audio_reviews.csv')
    
    # 如果已经处理过，直接加载
    if os.path.exists(processed_path):
        print(f"正在从 {processed_path} 加载已处理的数据")
        return pd.read_csv(processed_path)
    
    # 下载数据
    download_amazon_dataset(url, raw_path)
    
    # 加载数据
    df = load_amazon_dataset(raw_path)
    
    # 筛选音频设备相关评论
    audio_df = filter_audio_products(df)
    
    # 保存处理后的数据
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    audio_df.to_csv(processed_path, index=False)
    print(f"已将处理后的数据保存至 {processed_path}")
    
    return audio_df
```

## src/data/preprocessing.py

```python
import pandas as pd
import numpy as np
import re
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# 确保下载必要的NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text):
    """
    清洗文本数据
    
    Args:
        text (str): 原始文本
    
    Returns:
        str: 清洗后的文本
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # 转为小写
    text = text.lower()
    
    # 移除URL
    text = re.sub(r'http\S+', '', text)
    
    # 移除特殊字符，但保留标点
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
    
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text, remove_stopwords=True):
    """
    分词并可选地移除停用词
    
    Args:
        text (str): 文本
        remove_stopwords (bool): 是否移除停用词
    
    Returns:
        list: 分词列表
    """
    if not text:
        return []
    
    # 分词
    tokens = word_tokenize(text)
    
    # 移除停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.lower() not in stop_words]
    
    # 移除标点符号和数字
    tokens = [t for t in tokens if t.isalpha()]
    
    return tokens

def preprocess_reviews(df):
    """
    预处理评论数据
    
    Args:
        df (pd.DataFrame): 原始评论数据
    
    Returns:
        pd.DataFrame: 预处理后的数据
    """
    print("开始预处理...")
    processed_df = df.copy()
    
    # 重命名列以便更直观
    column_mapping = {
        'reviewerID': 'user_id',
        'asin': 'product_id',
        'reviewerName': 'user_name',
        'helpful': 'helpful_votes',
        'reviewText': 'review_text',
        'overall': 'rating',
        'summary': 'review_title',
        'unixReviewTime': 'review_timestamp',
        'reviewTime': 'review_date'
    }
    
    # 应用只有在数据集中存在的列的映射
    valid_mapping = {k: v for k, v in column_mapping.items() if k in processed_df.columns}
    processed_df = processed_df.rename(columns=valid_mapping)
    
    # 确保关键列存在
    required_columns = ['review_text', 'rating']
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    # 清洗文本
    print("清洗文本...")
    processed_df['clean_review_text'] = processed_df['review_text'].apply(clean_text)
    
    if 'review_title' in processed_df.columns:
        processed_df['clean_review_title'] = processed_df['review_title'].apply(clean_text)
    
    # 分词
    print("分词处理...")
    processed_df['tokens'] = processed_df['clean_review_text'].apply(tokenize_text)
    
    # 计算评论长度
    processed_df['review_length'] = processed_df['clean_review_text'].str.len()
    processed_df['word_count'] = processed_df['tokens'].apply(len)
    
    # 处理时间戳
    if 'review_timestamp' in processed_df.columns:
        processed_df['review_date'] = pd.to_datetime(processed_df['review_timestamp'], unit='s')
        processed_df['review_year'] = processed_df['review_date'].dt.year
        processed_df['review_month'] = processed_df['review_date'].dt.month
    
    # 处理helpful_votes字段
    if 'helpful_votes' in processed_df.columns and isinstance(processed_df['helpful_votes'].iloc[0], list):
        processed_df['helpful_count'] = processed_df['helpful_votes'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
        processed_df['total_votes'] = processed_df['helpful_votes'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 0)
    
    # 添加验证标记
    processed_df['verified_purchase'] = False  # 默认设为False，实际数据中可能有此字段
    
    print("预处理完成。")
    return processed_df

def create_price_categories(df, product_meta=None):
    """
    创建价格区间分类
    
    Args:
        df (pd.DataFrame): 预处理后的评论数据
        product_meta (pd.DataFrame, optional): 产品元数据，包含价格信息
    
    Returns:
        pd.DataFrame: 添加价格区间的数据
    """
    result_df = df.copy()
    
    if product_meta is not None and 'price' in product_meta.columns:
        # 如果有产品元数据，使用实际价格
        product_prices = product_meta[['asin', 'price']].rename(columns={'asin': 'product_id'})
        result_df = result_df.merge(product_prices, on='product_id', how='left')
        
        # 定义价格区间
        conditions = [
            (result_df['price'] < 50),
            (result_df['price'] >= 50) & (result_df['price'] < 150),
            (result_df['price'] >= 150) & (result_df['price'] < 300),
            (result_df['price'] >= 300)
        ]
        choices = ['预算型(<$50)', '中端($50-$150)', 
                   '高端($150-$300)', '豪华型(>$300)']
        
        result_df['price_range'] = np.select(conditions, choices, default='未知')
    else:
        # 如果没有产品元数据，使用产品ID的哈希值模拟
        import hashlib
        
        def assign_price_range(product_id):
            # 使用产品ID的哈希值模拟价格分布
            hash_value = int(hashlib.md5(str(product_id).encode()).hexdigest(), 16) % 4
            ranges = ['预算型(<$50)', '中端($50-$150)', 
                      '高端($150-$300)', '豪华型(>$300)']
            return ranges[hash_value]
        
        result_df['price_range'] = result_df['product_id'].apply(assign_price_range)
    
    return result_df

def assign_brand_categories(df, brand_mapping=None):
    """
    分配品牌类别
    
    Args:
        df (pd.DataFrame): 评论数据
        brand_mapping (dict, optional): 产品ID到品牌的映射
    
    Returns:
        pd.DataFrame: 添加品牌信息的数据
    """
    result_df = df.copy()
    
    if brand_mapping is not None:
        # 如果有品牌映射，直接使用
        result_df['brand'] = result_df['product_id'].map(brand_mapping)
    else:
        # 从评论文本中识别常见品牌
        common_brands = [
            'sony', 'bose', 'sennheiser', 'apple', 'beats', 'samsung', 
            'jabra', 'jbl', 'audio-technica', 'skullcandy', 'anker', 
            'soundcore', 'shure', 'akg', 'jaybird', 'plantronics', 'mpow'
        ]
        
        def extract_brand(text):
            if not isinstance(text, str):
                return 'unknown'
            
            text_lower = text.lower()
            for brand in common_brands:
                if brand in text_lower:
                    return brand
            return 'other'
        
        # 从评论标题和正文中提取品牌
        result_df['brand'] = result_df.apply(
            lambda row: extract_brand(str(row.get('clean_review_title', '')) + ' ' + str(row.get('clean_review_text', ''))), 
            axis=1
        )
    
    return result_df

def get_processed_dataset(raw_df=None, save_path='data/processed/processed_audio_reviews.csv'):
    """
    获取完整预处理的数据集
    
    Args:
        raw_df (pd.DataFrame, optional): 原始数据
        save_path (str): 处理后数据保存路径
    
    Returns:
        pd.DataFrame: 完整预处理的数据集
    """
    # 如果保存路径存在，直接加载
    try:
        processed_df = pd.read_csv(save_path)
        print(f"从 {save_path} 加载已处理的数据集")
        return processed_df
    except FileNotFoundError:
        pass
    
    if raw_df is None:
        raise ValueError("未提供原始数据且未找到已处理数据。")
    
    # 执行预处理
    df = preprocess_reviews(raw_df)
    
    # 创建价格区间
    df = create_price_categories(df)
    
    # 分配品牌
    df = assign_brand_categories(df)
    
    # 保存处理后的数据
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"已将处理后的数据集保存至 {save_path}")
    
    return df
```

## src/features/aspect_extraction.py

```python
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
```

## src/visualization/plots.py

```python
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
```

## src/utils/export.py

```python
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

def clean_for_excel(df):
    """
    清理DataFrame中的非法Excel字符
    
    Args:
        df (pd.DataFrame): 原始数据框
        
    Returns:
        pd.DataFrame: 清理后的数据框
    """
    # 复制DataFrame避免修改原始数据
    cleaned_df = df.copy()
    
    # 定义Excel不支持的字符的正则表达式
    illegal_chars_regex = r'[\000-\010]|[\013-\014]|[\016-\037]'
    
    # 对每个字符串类型的列进行清理
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # 对字符串类型的列应用替换
            cleaned_df[col] = cleaned_df[col].astype(str).apply(
                lambda x: re.sub(illegal_chars_regex, '', x) if pd.notnull(x) else x
            )
            
            # 处理其他特殊字符
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str(x)) if pd.notnull(x) else x
            )
    
    return cleaned_df

def prepare_powerbi_data(df, output_path='outputs/powerbi_data'):
    """
    为 PowerBI 准备数据文件
    
    Args:
        df (pd.DataFrame): 带有方面分析的评论数据
        output_path (str): 输出目录路径
    
    Returns:
        dict: 包含各个数据文件路径的字典
    """
    print("正在准备 PowerBI 数据...")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 获取方面评分列
    aspect_cols = [col for col in df.columns if col.endswith('_score')]
    aspects = [col.replace('_score', '') for col in aspect_cols]
    
    # 1. 主数据表 - 这将是我们的事实表
    main_data = df.copy()
    
    # 将方面评分数据转换为长格式，更适合PowerBI处理
    aspect_data_list = []
    
    for _, row in df.iterrows():
        review_id = row.get('review_id', '') or row.get('user_id', '') or ''
        product_id = row.get('product_id', '')
        rating = row.get('rating', 0)
        brand = row.get('brand', 'unknown')
        price_range = row.get('price_range', 'unknown')
        review_date = row.get('review_date', None)
        
        # 对每个方面创建一行
        for aspect in aspects:
            score_col = f"{aspect}_score"
            count_col = f"{aspect}_count"
            
            if score_col in row and not pd.isna(row[score_col]):
                aspect_data_list.append({
                    'review_id': review_id,
                    'product_id': product_id,
                    'aspect': aspect,
                    'score': row[score_col],
                    'count': row.get(count_col, 0),
                    'rating': rating,
                    'brand': brand,
                    'price_range': price_range,
                    'review_date': review_date
                })
    
    aspect_data = pd.DataFrame(aspect_data_list)
    
    # 2. 品牌维度表
    if 'brand' in df.columns:
        brands = df['brand'].unique()
        brand_data = pd.DataFrame({
            'brand': brands,
            'review_count': [len(df[df['brand'] == b]) for b in brands]
        })
        
        # 计算每个品牌的平均评分
        brand_data['avg_rating'] = [df[df['brand'] == b]['rating'].mean() for b in brands]
        
        # 添加每个品牌的最高评分方面
        brand_best_aspects = []
        for brand in brands:
            brand_df = df[df['brand'] == brand]
            best_aspect = ''
            best_score = 0
            for aspect in aspects:
                score_col = f"{aspect}_score"
                if score_col in df.columns:
                    avg_score = brand_df[score_col].mean()
                    if avg_score > best_score:
                        best_score = avg_score
                        best_aspect = aspect
            brand_best_aspects.append({
                'brand': brand,
                'best_aspect': best_aspect,
                'best_aspect_score': best_score
            })
        brand_best_aspects_df = pd.DataFrame(brand_best_aspects)
        brand_data = pd.merge(brand_data, brand_best_aspects_df, on='brand', how='left')
    else:
        brand_data = pd.DataFrame(columns=['brand', 'review_count', 'avg_rating'])
    
    # 3. 价格区间维度表
    if 'price_range' in df.columns:
        price_ranges = df['price_range'].unique()
        price_data = pd.DataFrame({
            'price_range': price_ranges,
            'review_count': [len(df[df['price_range'] == p]) for p in price_ranges]
        })
        
        # 计算每个价格区间的平均评分
        price_data['avg_rating'] = [df[df['price_range'] == p]['rating'].mean() for p in price_ranges]
    else:
        price_data = pd.DataFrame(columns=['price_range', 'review_count', 'avg_rating'])
    
    # 4. 方面维度表
    aspect_info = []
    for aspect in aspects:
        score_col = f"{aspect}_score"
        if score_col in df.columns:
            avg_score = df[score_col].dropna().mean()
            count = df[score_col].count()
            aspect_info.append({
                'aspect': aspect,
                'aspect_name': aspect.replace('_', ' ').title(),
                'avg_score': avg_score,
                'count': count
            })
    aspect_dim = pd.DataFrame(aspect_info)
    
    # 5. 时间维度表(如果有日期数据)
    if 'review_date' in df.columns and 'review_date' in aspect_data.columns:
        # 确保日期列是datetime类型
        try:
            aspect_data['review_date'] = pd.to_datetime(aspect_data['review_date'])
            aspect_data['year'] = aspect_data['review_date'].dt.year
            aspect_data['month'] = aspect_data['review_date'].dt.month
            aspect_data['quarter'] = aspect_data['review_date'].dt.quarter
            aspect_data['year_month'] = aspect_data['review_date'].dt.strftime('%Y-%m')
        except:
            print("无法处理日期数据，跳过时间维度表创建")
    
    # 保存数据文件
    main_path = os.path.join(output_path, 'main_data.csv')
    aspect_path = os.path.join(output_path, 'aspect_data.csv')
    brand_path = os.path.join(output_path, 'brand_data.csv')
    price_path = os.path.join(output_path, 'price_data.csv')
    aspect_dim_path = os.path.join(output_path, 'aspect_dim.csv')
    
    main_data.to_csv(main_path, index=False)
    aspect_data.to_csv(aspect_path, index=False)
    brand_data.to_csv(brand_path, index=False)
    price_data.to_csv(price_path, index=False)
    aspect_dim.to_csv(aspect_dim_path, index=False)
    
    print(f"数据已成功导出至 {output_path} 目录")
    
    # 同时导出一个整合的Excel文件，方便直接导入PowerBI
    excel_path = os.path.join(output_path, 'powerbi_data.xlsx')
    
    # 在导出到Excel前清理数据
    clean_main_data = clean_for_excel(main_data)
    clean_aspect_data = clean_for_excel(aspect_data)
    clean_brand_data = clean_for_excel(brand_data)
    clean_price_data = clean_for_excel(price_data)
    clean_aspect_dim = clean_for_excel(aspect_dim)
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            clean_main_data.to_excel(writer, sheet_name='主数据', index=False)
            clean_aspect_data.to_excel(writer, sheet_name='方面评分数据', index=False)
            clean_brand_data.to_excel(writer, sheet_name='品牌数据', index=False)
            clean_price_data.to_excel(writer, sheet_name='价格区间数据', index=False)
            clean_aspect_dim.to_excel(writer, sheet_name='方面维度', index=False)
        
        print(f"整合的Excel文件已保存至 {excel_path}")
    except Exception as e:
        print(f"导出Excel文件时发生错误: {e}")
        print("但CSV文件已成功导出，您可以直接将CSV文件导入PowerBI")
    
    return {
        'main_data': main_path,
        'aspect_data': aspect_path,
        'brand_data': brand_path,
        'price_data': price_path,
        'aspect_dim': aspect_dim_path,
        'excel_file': excel_path if os.path.exists(excel_path) else None
    }
```

## main.py

```python
import os
import pandas as pd
import argparse
from src.data.acquisition import get_audio_dataset
from src.data.preprocessing import get_processed_dataset
from src.features.aspect_extraction import create_review_aspects_dataset
from src.visualization.plots import create_all_visualizations
from src.utils.export import prepare_powerbi_data

def main(args):
    """
    主程序函数，运行完整分析流程
    
    Args:
        args: 命令行参数
    
    Returns:
        None
    """
    print("开始音频评论分析项目...")
    
    # 创建必要的目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/powerbi_data', exist_ok=True)
    
    # 控制执行的步骤
    do_data_collection = args.all or args.data_collection
    do_preprocessing = args.all or args.preprocessing
    do_aspect_analysis = args.all or args.aspect_analysis
    do_visualization = args.all or args.visualization
    
    # 1. 数据获取
    raw_data = None
    if do_data_collection:
        print("\n=== 步骤 1: 数据获取 ===")
        raw_data = get_audio_dataset(
            url=args.data_url,
            save_dir='data/raw',
            processed_dir='data/processed'
        )
    
    # 2. 数据预处理
    processed_data = None
    if do_preprocessing:
        print("\n=== 步骤 2: 数据预处理 ===")
        # 如果前一步没有加载数据，尝试从文件加载
        if raw_data is None:
            try:
                raw_data = pd.read_csv('data/processed/audio_reviews.csv')
                print("从文件加载原始数据。")
            except FileNotFoundError:
                print("未找到原始数据文件。请先运行 --data-collection 步骤。")
                return
        
        processed_data = get_processed_dataset(
            raw_df=raw_data,
            save_path='data/processed/processed_audio_reviews.csv'
        )
    
    # 3. 方面分析
    aspect_data = None
    if do_aspect_analysis:
        print("\n=== 步骤 3: 方面分析 ===")
        # 如果前一步没有加载数据，尝试从文件加载
        if processed_data is None:
            try:
                processed_data = pd.read_csv('data/processed/processed_audio_reviews.csv')
                print("从文件加载预处理数据。")
            except FileNotFoundError:
                print("未找到预处理数据文件。请先运行 --preprocessing 步骤。")
                return
        
        aspect_data = create_review_aspects_dataset(
            df=processed_data,
            output_path='data/processed/review_aspects.csv'
        )
    
    # 4. 可视化
    if do_visualization:
        print("\n=== 步骤 4: 可视化 ===")
        # 如果前一步没有加载数据，尝试从文件加载
        if aspect_data is None:
            try:
                aspect_data = pd.read_csv('data/processed/review_aspects.csv')
                print("从文件加载方面分析数据。")
            except FileNotFoundError:
                print("未找到方面分析数据文件。请先运行 --aspect-analysis 步骤。")
                return
        
        create_all_visualizations(
            df=aspect_data,
            output_dir='outputs/figures'
        )
    
    # 5. 导出PowerBI数据（可选）
    if args.export_powerbi:
        print("\n=== 步骤 5: 导出PowerBI数据 ===")
        # 如果前一步没有加载数据，尝试从文件加载
        if aspect_data is None:
            try:
                aspect_data = pd.read_csv('data/processed/review_aspects.csv')
                print("从文件加载方面分析数据。")
            except FileNotFoundError:
                print("未找到方面分析数据文件。请先运行 --aspect-analysis 步骤。")
                return
        
        # 导出数据以便在PowerBI中使用
        prepare_powerbi_data(
            df=aspect_data,
            output_path='outputs/powerbi_data'
        )
    
    print("\n音频评论分析项目成功完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='音频设备评论分析')
    
    parser.add_argument('--all', action='store_true', 
                        help='运行所有步骤')
    parser.add_argument('--data-collection', action='store_true', 
                        help='运行数据获取步骤')
    parser.add_argument('--preprocessing', action='store_true', 
                        help='运行预处理步骤')
    parser.add_argument('--aspect-analysis', action='store_true', 
                        help='运行方面分析步骤')
    parser.add_argument('--visualization', action='store_true', 
                        help='运行可视化步骤')
    parser.add_argument('--export-powerbi', action='store_true', 
                        help='导出数据用于PowerBI可视化')
    parser.add_argument('--data-url', type=str, 
                        default="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
                        help='数据集URL')
    
    args = parser.parse_args()
    
    # 如果没有指定任何步骤，默认运行所有步骤
    if not any([args.all, args.data_collection, args.preprocessing, 
                args.aspect_analysis, args.visualization, args.export_powerbi]):
        args.all = True
    
    main(args)
```

## dashboard.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os
import sys

# 设置页面配置
st.set_page_config(
    page_title="音频设备评论分析仪表盘",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题和介绍
st.title("🎧 音频设备评论分析仪表盘")
st.markdown("""
这个仪表盘展示了从Amazon评论中提取的音频设备评价分析，包括各方面评分、品牌比较和价格区间分析。
""")

# 从CSV文件加载数据
@st.cache_data
def load_data():
    """加载主数据集和方面数据"""
    try:
        df = pd.read_csv('data/processed/review_aspects.csv')
        aspect_data = pd.read_csv('outputs/powerbi_data/aspect_data.csv')
        brand_data = pd.read_csv('outputs/powerbi_data/brand_data.csv')
        price_data = pd.read_csv('outputs/powerbi_data/price_data.csv')
        aspect_dim = pd.read_csv('outputs/powerbi_data/aspect_dim.csv')
        return df, aspect_data, brand_data, price_data, aspect_dim
    except FileNotFoundError:
        st.error("数据文件未找到。请确保已运行数据处理步骤。")
        if not os.path.exists('data/processed/review_aspects.csv'):
            st.info("尝试运行 `python main.py --all` 生成所需数据文件")
        return None, None, None, None, None

# 加载数据
df, aspect_data, brand_data, price_data, aspect_dim = load_data()

# 检查数据是否成功加载
if df is None:
    st.stop()

# 侧边栏 - 筛选器
st.sidebar.header("数据筛选")

# 品牌筛选
all_brands = sorted(df['brand'].unique().tolist())
selected_brands = st.sidebar.multiselect(
    "选择品牌", 
    options=all_brands,
    default=all_brands[:5] if len(all_brands) > 5 else all_brands
)

# 价格区间筛选
all_price_ranges = sorted(df['price_range'].unique().tolist())
selected_price_ranges = st.sidebar.multiselect(
    "选择价格区间", 
    options=all_price_ranges,
    default=all_price_ranges
)

# 评分范围筛选
min_rating, max_rating = st.sidebar.slider(
    "评分范围", 
    min_value=1.0, 
    max_value=5.0,
    value=(1.0, 5.0),
    step=0.5
)

# 筛选数据
filtered_df = df.copy()
if selected_brands:
    filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
if selected_price_ranges:
    filtered_df = filtered_df[filtered_df['price_range'].isin(selected_price_ranges)]
filtered_df = filtered_df[(filtered_df['rating'] >= min_rating) & (filtered_df['rating'] <= max_rating)]

# 创建标签页
tab1, tab2, tab3, tab4 = st.tabs(["总体概览", "方面分析", "品牌比较", "价格区间分析"])

with tab1:
    st.header("总体概览")
    
    # 行1: 关键指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("评论总数", len(filtered_df))
    with col2:
        st.metric("平均评分", f"{filtered_df['rating'].mean():.2f}")
    with col3:
        st.metric("品牌数量", len(filtered_df['brand'].unique()))
    with col4:
        st.metric("价格区间数量", len(filtered_df['price_range'].unique()))
    
    # 行2: 评分分布和词云
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("评分分布")
        rating_counts = filtered_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, 
            y=rating_counts.values,
            labels={'x': '评分', 'y': '数量'},
            title='评分分布',
            color=rating_counts.index,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("评论字数分布")
        fig = px.histogram(
            filtered_df, 
            x="word_count",
            nbins=50,
            labels={'word_count': '字数', 'count': '评论数量'},
            title='评论字数分布'
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("方面分析")
    
    # 获取方面评分列
    aspect_cols = [col for col in filtered_df.columns if col.endswith('_score')]
    aspects = [col.replace('_score', '') for col in aspect_cols]
    
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
    
    # 行1: 方面选择
    selected_aspect = st.selectbox(
        "选择要详细分析的方面",
        options=aspects,
        format_func=lambda x: aspect_translation.get(x, x)
    )
    
    # 行2: 方面评分雷达图和详细分析
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("各方面平均评分")
        # 计算每个方面的平均得分
        aspect_means = {}
        for aspect, aspect_col in zip(aspects, aspect_cols):
            aspect_means[aspect] = filtered_df[aspect_col].mean()
        
        # 排序并整理数据
        aspect_means = {k: v for k, v in sorted(aspect_means.items(), key=lambda item: item[1], reverse=True)}
        
        # 雷达图数据
        categories = [aspect_translation.get(k, k) for k in aspect_means.keys()]
        values = list(aspect_means.values())
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
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"{aspect_translation.get(selected_aspect, selected_aspect)}详细分析")
        score_col = f"{selected_aspect}_score"
        
        # 评分分布
        fig = px.histogram(
            filtered_df[filtered_df[score_col].notna()], 
            x=score_col,
            nbins=20,
            labels={score_col: '评分', 'count': '数量'},
            title=f'{aspect_translation.get(selected_aspect, selected_aspect)}评分分布'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 平均评分
        avg_score = filtered_df[score_col].mean()
        st.metric(f"{aspect_translation.get(selected_aspect, selected_aspect)}平均评分", f"{avg_score:.2f}/10")

with tab3:
    st.header("品牌比较")
    
    # 获取评论数量足够的品牌
    brand_review_counts = filtered_df['brand'].value_counts()
    min_reviews = st.slider("最少评论数量", min_value=5, max_value=100, value=30)
    valid_brands = brand_review_counts[brand_review_counts >= min_reviews].index.tolist()
    
    if not valid_brands:
        st.warning(f"没有品牌的评论数量至少为 {min_reviews}。请降低最少评论数量阈值。")
    else:
        # 最多显示10个品牌
        valid_brands = valid_brands[:10]
        
        # 为每个品牌计算平均方面评分
        brand_data = []
        
        for brand in valid_brands:
            brand_df = filtered_df[filtered_df['brand'] == brand]
            
            # 计算这个品牌的方面平均分
            for aspect in aspects:
                score_col = f"{aspect}_score"
                avg_score = brand_df[score_col].mean()
                if not np.isnan(avg_score):
                    brand_data.append({
                        'Brand': brand,
                        'Aspect': aspect_translation.get(aspect, aspect),
                        'Score': avg_score
                    })
        
        # 创建数据框
        brand_scores_df = pd.DataFrame(brand_data)
        
        if len(brand_scores_df) > 0:
            # 创建热力图
            heatmap_data = brand_scores_df.pivot(index='Brand', columns='Aspect', values='Score')
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
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 品牌总体评分条形图
            st.subheader("品牌总体评分")
            brand_overall = heatmap_data.mean(axis=1).sort_values(ascending=False)
            fig = px.bar(
                x=brand_overall.index,
                y=brand_overall.values,
                labels={'x': '品牌', 'y': '平均评分'},
                color=brand_overall.values,
                color_continuous_scale='Viridis',
                range_color=[1, 10]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("没有足够的数据进行品牌比较。")

with tab4:
    st.header("价格区间分析")
    
    # 为每个价格区间计算方面评分
    price_range_data = []
    
    for price_range in selected_price_ranges:
        price_df = filtered_df[filtered_df['price_range'] == price_range]
        
        if len(price_df) < 10:  # 跳过样本太少的价格区间
            continue
            
        # 计算这个价格区间的方面平均分
        for aspect in aspects:
            score_col = f"{aspect}_score"
            avg_score = price_df[score_col].mean()
            if not np.isnan(avg_score):
                price_range_data.append({
                    'Price Range': price_range,
                    'Aspect': aspect_translation.get(aspect, aspect),
                    'Score': avg_score
                })
    
    # 创建数据框
    price_scores_df = pd.DataFrame(price_range_data)
    
    if len(price_scores_df) > 0:
        # 创建热力图
        heatmap_data = price_scores_df.pivot(index='Price Range', columns='Aspect', values='Score')
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="方面", y="价格区间", color="分数"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            aspect="auto",
            color_continuous_scale='Viridis',
            range_color=[1, 10],
            title='价格区间方面评分比较'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 价格区间折线图
        st.subheader("不同价格区间各方面得分比较")
        fig = go.Figure()
        
        for aspect in heatmap_data.columns:
            fig.add_trace(go.Scatter(
                x=heatmap_data.index,
                y=heatmap_data[aspect],
                mode='lines+markers',
                name=aspect
            ))
        
        fig.update_layout(
            xaxis_title="价格区间",
            yaxis_title="平均分数 (1-10)",
            yaxis=dict(range=[0, 10])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 价值比(性价比)分析
        st.subheader("性价比分析")
        
        # 简单定义性价比 = 音质得分 / 相对价格指数
        price_value_map = {
            '预算型(<$50)': 1,
            '中端($50-$150)': 2,
            '高端($150-$300)': 3,
            '豪华型(>$300)': 4
        }
        
        if '音质' in heatmap_data.columns and all(pr in price_value_map for pr in heatmap_data.index):
            value_data = []
            for pr in heatmap_data.index:
                score = heatmap_data.loc[pr, '音质']
                price_index = price_value_map.get(pr, 1)
                value_ratio = score / price_index
                value_data.append({
                    'Price Range': pr,
                    'Sound Quality': score,
                    'Value Ratio': value_ratio
                })
            
            value_df = pd.DataFrame(value_data)
            fig = px.bar(
                value_df,
                x='Price Range',
                y='Value Ratio',
                color='Sound Quality',
                labels={'Value Ratio': '性价比指数', 'Price Range': '价格区间'},
                title='音质性价比分析',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("没有足够的数据进行价格区间分析。")

# 底部信息
st.markdown("---")
st.markdown("© 2025 音频设备评论分析系统 | 使用Streamlit创建")
```

## requirements.txt

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
nltk>=3.6.0
spacy>=3.1.0
scikit-learn>=1.0.0
wordcloud>=1.8.0
tqdm>=4.60.0
requests>=2.25.0
openpyxl>=3.0.0
streamlit>=1.8.0
```

## 文件结构创建脚本 (setup.sh)

```bash
#!/bin/bash

# 创建项目目录结构
echo "创建项目目录结构..."
mkdir -p data/raw data/processed
mkdir -p outputs/figures outputs/powerbi_data
mkdir -p src/data src/features src/visualization src/utils

# 创建空的__init__.py文件
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 下载NLTK资源
echo "下载NLTK资源..."
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"

# 下载spaCy模型
echo "下载spaCy模型..."
python -m spacy download en_core_web_sm

echo "项目环境已准备就绪！"
echo "运行 'python main.py --all' 开始数据分析流程"
echo "运行 'streamlit run dashboard.py' 启动交互式仪表盘"
```

## 使用指南

### 安装与设置

1. **安装依赖**:
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt vader_lexicon stopwords
python -m spacy download en_core_web_sm
```

2. **创建目录结构**:
```bash
chmod +x setup.sh
./setup.sh
```

### 运行分析

```bash
# 运行所有步骤
python main.py --all

# 分步骤运行
python main.py --data-collection  # 获取数据
python main.py --preprocessing     # 数据预处理
python main.py --aspect-analysis   # 方面分析
python main.py --visualization     # 可视化
python main.py --export-powerbi    # 导出数据

# 启动Streamlit仪表盘
streamlit run dashboard.py
```

以上是完整的音频设备评论分析系统代码，包括了所有模块、PowerBI数据导出功能和新添加的Streamlit仪表盘。请根据需要自行调整参数和配置。
## 📊 数据分析方法

### 文本分析流程

1. **文本预处理**
   - 转为小写
   - 移除URL和特殊字符
   - 分词
   - 移除停用词

2. ## 方法论 - 方面提取 (Aspect Extraction)

本项目核心的方面提取功能旨在从用户评论中识别出讨论的产品特性（方面），并将评论的情感倾向与之关联。

### 当前实现方法

当前版本采用了一种基于 **预定义关键词匹配** 的方法，结合 **评论整体评分** 作为情感代理。具体步骤如下：

1.  **预定义方面与关键词:** 在 `config.py` 文件中，我们预先定义了一组常见的音频产品方面（如 "Sound Quality", "Battery Life", "Comfort" 等），并为每个方面关联了一系列相关的关键词列表。
2.  **文本预处理:** 评论文本经过基础预处理，包括转为小写、去除标点符号等（详见 `src/data/preprocessing.py`）。
3.  **关键词匹配:** 系统遍历每一条预处理后的评论文本。对于 `config.py` 中定义的每一个方面，系统检查评论文本中是否包含该方面对应的**任何一个关键词**（当前实现为子字符串匹配）。
4.  **方面识别:** 如果评论文本中找到了某一方面的一个或多个关键词，则认为该评论提及了该方面。
5.  **情感/评分关联:** **关键假设：** 当前方法将该条评论的**整体评分 (rating)** 直接作为其所提及的所有方面的情感得分。例如，一条评分为 5 星且提及了 "sound" 和 "comfort" 的评论，会被记录为 "Sound Quality" 得分 5，"Comfort" 得分 5。

### 方法的优点

* **实现简单:** 该方法逻辑清晰，代码实现相对直接。
* **计算快速:** 纯文本匹配操作，对于中等规模的数据集处理速度较快。
* **可解释性强:** 提取出的方面直接基于明确的关键词，易于理解为何某条评论被归类到特定方面。
* **易于定制:** 通过修改 `config.py` 中的关键词列表，可以方便地调整或扩展覆盖的方面。

### 方法的局限性与挑战

尽管该方法在当前阶段有效，但也存在一些明显的局限性：

* **上下文理解不足:** 无法理解词语的实际含义和上下文。例如，无法区分 "good sound" 和 "no sound"。
* **忽略否定词和修饰词:** 简单的关键词匹配不能处理否定情况（如 "not comfortable" 可能会因为匹配到 "comfortable" 而被错误识别）或程度副词（如 "very loud" 和 "slightly loud"）。
* **情感关联粗糙:** 使用评论的整体评分作为每个方面的情感代理是一个**强假设**。一条评论可能同时称赞音质（正面）但抱怨电池续航（负面），整体评分可能无法准确反映对单个方面的情感。
* **关键词依赖:** 效果高度依赖于 `config.py` 中关键词列表的**质量和完备性**。未包含的关键词或新的表达方式将无法被识别。
* **多方面情感混淆:** 难以区分评论中针对不同方面的不同情感表达。

### 未来工作与潜在改进方向

为了克服当前方法的局限性，未来可以探索以下改进方向：

1.  **改进关键词匹配:**
    * 使用**全词匹配**（例如正则表达式 `\bkeyword\b`）避免部分匹配带来的歧义。
    * 结合**词形还原 (Lemmatization)** 或 **词干提取 (Stemming)** 将不同形态的词（如 "connect", "connecting", "connection"）归一化。
2.  **引入规则和情感词典:**
    * 在关键词附近查找**情感词**（如 "good", "bad", "amazing", "terrible"）。
    * 考虑情感词前的**否定词**（"not good"）和**程度副词**（"very good"）来调整情感评分。可以使用现有的情感词典库（如 VADER, SentiWordNet）。
3.  **基于依赖关系解析:**
    * 使用 `spaCy` 等库进行**依存句法分析**，找出与方面关键词直接相关的评价性词语，建立更准确的方面-情感链接。
4.  **机器学习/深度学习方法:**
    * **方面术语抽取 (ATE):** 使用序列标注模型（如 BiLSTM-CRF, BERT）自动从文本中抽取出方面词语，减少对预定义关键词的依赖。
    * **方面情感分类 (ASC):** 对于已识别出的方面词，使用分类模型（如 BERT-based classifiers）判断其在该上下文中的具体情感极性（正面/负面/中性）和强度。这可以解决整体评分带来的问题。
    * **端到端 ABSA:** 使用统一的模型同时完成方面抽取和情感分类任务。


3. **情感分析**
   - 使用NLTK的VADER情感分析器
   - 计算每个方面相关句子的情感得分
   - 规范化得分到1-10范围

4. **品牌识别**
   - 从评论文本中提取常见音频设备品牌
   - 支持通过外部映射提供产品ID到品牌的映射

5. **价格区间分类**
   - 支持实际价格数据
   - 在无实际价格数据时使用哈希值模拟价格分布

## 📈 可视化结果

系统生成以下可视化：

1. **评分分布**：展示评分频率分布
2. **方面评分雷达图**：直观显示各方面的平均得分
3. **品牌比较热力图**：比较不同品牌在各方面的表现
4. **价格区间比较**：分析不同价格区间产品的性价比
5. **高低评分词云**：展示高评分和低评分评论中的关键词

## 🚀 使用指南

### 命令行参数

```bash
# 运行所有步骤
python main.py --all

# 分步骤运行
python main.py --data-collection  # 获取数据
python main.py --preprocessing     # 数据预处理
python main.py --aspect-analysis   # 方面分析
python main.py --visualization     # 可视化
python main.py --export-powerbi    # 导出PowerBI数据

# 使用自定义数据源
python main.py --data-url "http://your-data-source.com/dataset.json.gz"
```

### 自定义分析

要自定义分析，可以修改配置文件中的参数：

- 在`AUDIO_KEYWORDS`中添加或删除关键词以更改音频设备筛选条件
- 在`ASPECT_CONFIG`中修改方面定义及其关键词
- 在`VIZ_CONFIG`中调整可视化参数

## 🔧 扩展与优化

### 潜在改进方向

1. **更高级的方面提取**：
   - 使用依存句法分析提高方面识别准确性
   - 实现基于BERT或其他预训练模型的方面提取

2. **更细粒度的情感分析**：
   - 区分方面特定的情感极性
   - 识别评论中的比较和对比表达

3. **整合外部数据**：
   - 添加产品规格数据
   - 添加时间序列分析，跟踪产品评价随时间变化

4. **交互式可视化**：
   - 开发基于Web的交互式仪表板
   - 支持实时数据更新和筛选

## 📚 参考资料与数据来源

- 数据集：Amazon电子产品评论数据集
- 自然语言处理库：NLTK、spaCy
- 可视化库：Matplotlib、Seaborn、Plotly

## 🧠 技术原理

### 方面提取与情感分析

本项目使用基于关键词的方法从评论中提取方面信息，并使用VADER情感分析器计算情感得分。VADER是一个基于规则的情感分析模型，专为社交媒体文本设计，能够处理俚语、表情符号和常见缩写。

### 规范化得分计算

为了将情感得分转换为直观的1-10评分尺度，系统使用以下转换公式：

```
normalized_score = ((raw_sentiment_score + 1) / 2) * 9 + 1
```

这将原始的-1到1的情感得分映射到1-10的范围内。

## 结语

音频设备评论分析系统是一个功能完善的数据分析工具，它展示了如何应用自然语言处理技术从非结构化文本数据中提取有价值的见解。通过对用户评论的深入分析，该系统能够帮助音频设备制造商、营销人员和消费者更好地了解产品的优缺点，指导产品改进和购买决策。

音频设备评论分析系统采用模块化设计，易于扩展和定制。您可以根据自己的需求修改配置参数，添加新的分析维度，或者集成其他数据源，以满足特定的分析需求。
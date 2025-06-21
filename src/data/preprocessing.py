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
    nltk.data.find('tokenizers/punkt_tab/english')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
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
    
    # 处理helpful_votes字段（可能为列表或字符串形式的列表）
    if 'helpful_votes' in processed_df.columns:
        def parse_helpful(x):
            if isinstance(x, list):
                return x[0] if len(x) > 0 else 0, x[1] if len(x) > 1 else 0
            try:
                votes = eval(x) if isinstance(x, str) else x
                return votes[0] if len(votes) > 0 else 0, votes[1] if len(votes) > 1 else 0
            except:
                return 0, 0
                
        helpful_data = processed_df['helpful_votes'].apply(parse_helpful)
        processed_df['helpful_count'] = helpful_data.apply(lambda x: x[0])
        processed_df['total_votes'] = helpful_data.apply(lambda x: x[1])
    
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

def get_processed_dataset(raw_df=None, save_path='data/processed/processed_audio_reviews.csv', max_reviews=5000):
    """
    获取完整预处理的数据集
    
    Args:
        raw_df (pd.DataFrame, optional): 原始数据
        save_path (str): 处理后数据保存路径
        max_reviews (int): 最大评论数量限制
    
    Returns:
        pd.DataFrame: 完整预处理的数据集
    """
    # 修正后的逻辑：首先尝试加载已存在的文件
    try:
        # 如果保存路径存在，直接加载
        processed_df = pd.read_csv(save_path)
        print(f"从 {save_path} 加载已处理的数据集，跳过预处理步骤。")
        # 如果已有数据超过限制，则随机采样
        if len(processed_df) > max_reviews:
            processed_df = processed_df.sample(n=max_reviews, random_state=42)
            print(f"随机选取 {max_reviews} 条评论")
        return processed_df
    except FileNotFoundError:
        # 这是首次运行时的正常路径
        print(f"未找到已处理的数据文件 '{save_path}'。将从原始数据开始生成。")
        # `pass` 将让函数继续往下执行，而不是报错退出
        pass
    except Exception as e:
        # 捕获其他可能的加载错误，例如文件损坏
        print(f"加载已有的 '{save_path}' 文件时发生错误: {e}")
        print("将尝试从原始数据重新生成。")
        pass

    # 如果未能从文件加载，则继续执行处理流程
    if raw_df is None:
        # 这个错误只应该在既没有找到缓存文件，又没有提供原始数据时触发
        raise ValueError("错误：未提供原始数据DataFrame，且未找到已缓存的处理后数据文件。")
    
    # 如果原始数据超过限制，先随机采样
    if len(raw_df) > max_reviews:
        raw_df = raw_df.sample(n=max_reviews, random_state=42)
        print(f"随机选取 {max_reviews} 条评论进行预处理")
    
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

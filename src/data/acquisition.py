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

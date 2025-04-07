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

# 使用示例
if __name__ == "__main__":
    # 加载处理后的数据
    try:
        df = pd.read_csv('data/processed/review_aspects.csv')
        prepare_powerbi_data(df)
    except FileNotFoundError:
        print("未找到处理后的数据文件，请先运行数据处理步骤")
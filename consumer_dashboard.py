import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import pandas as pd
import numpy as np
import datetime
import re
from collections import Counter
from collections import defaultdict  # 添加缺失的导入
from functools import lru_cache

# 配置Streamlit页面
st.set_page_config(
    page_title="音频设备评论分析仪表盘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 数据缓存
@st.cache_data(ttl=3600, show_spinner=False)
def load_base_data():
    # 加载实际分析结果
    try:
        # 尝试分块读取CSV文件以避免内存问题
        chunks = pd.read_csv('data/processed/review_aspects.csv', 
                           chunksize=10000,
                           dtype={
                               'rating': 'float64',
                               'helpful_count': 'float64',
                               'review_timestamp': 'float64'
                           })
        
        aspect_df = pd.concat(chunks)
        
        # 确保所有_score列都是数值类型
        for col in aspect_df.columns:
            if '_score' in col:
                aspect_df[col] = pd.to_numeric(aspect_df[col], errors='coerce')
                
        # 验证必要列是否存在且类型正确
        required_cols = {
            'rating': 'float64',
            'review_text': 'object',
            'review_timestamp': 'float64'
        }
        
        missing_cols = [col for col in required_cols if col not in aspect_df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
            
        # 创建日期列
        aspect_df['review_date'] = pd.to_datetime(aspect_df['review_timestamp'], unit='s')
        
        # 验证数据有效性
        if aspect_df['rating'].isna().all():
            raise ValueError("评分数据无效")
        product_info = {
            "title": "深海声学 X1 无线降噪耳机 - 幻夜黑",
            "image_url": "https://via.placeholder.com/300x300.png?text=耳机图片",
            "overall_rating": aspect_df['rating'].mean(),
            "total_reviews": len(aspect_df),
            "price": "¥499.00"
        }
        
        # 转换实际数据为展示格式
        reviews = []
        for _, row in aspect_df.iterrows():
            # 提取最重要的方面
            aspects = {}
            aspect_scores = {}  # 保存原始分数用于排序
            
            for col in row.index:
                if col.endswith('_score') and not pd.isna(row[col]):
                    aspect = col.replace('_score', '')
                    try:
                        score = float(row[col])
                        aspects[aspect] = "positive" if score >= 5 else "negative"
                        aspect_scores[aspect] = score  # 保存原始分数
                    except (ValueError, TypeError):
                        continue
            
            # 只有在有分数时进行排序
            if aspect_scores:
                sorted_aspects = sorted(aspect_scores.items(), key=lambda x: abs(x[1]-5), reverse=True)[:3]
                aspects = {aspect: aspects[aspect] for aspect, _ in sorted_aspects}
            
            reviews.append({
                "id": row.name,
                "reviewer_name": f"用户{row.name}",
                "rating": row['rating'],
                "title": row.get('review_title', ''),
                "date": pd.to_datetime(row['review_timestamp'], unit='s').date(),
                "verified": True,
                "text": row['review_text'],
                "helpful_votes": row.get('helpful_count', 0),
                "aspects": aspects
            })
        
        return product_info, reviews
    
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        # 返回空数据但保持结构
        return {
            "title": "音频设备",
            "image_url": "https://via.placeholder.com/300x300.png?text=产品图片",
            "overall_rating": 0,
            "total_reviews": 0,
            "price": "¥0.00"
        }, []

# 获取基础数据并验证
try:
    product_info, all_reviews = load_base_data()
except Exception as e:
    st.error(f"无法加载基础数据: {str(e)}")
    product_info = {
        "title": "音频设备",
        "image_url": "https://via.placeholder.com/300x300.png?text=产品图片",
        "overall_rating": 0,
        "total_reviews": 0,
        "price": "¥0.00"
    }
    all_reviews = []

if not product_info or not all_reviews:
    st.warning("正在使用演示数据，请先运行分析流程生成完整数据")
    # 添加运行分析按钮
    if st.button("运行分析流程"):
        with st.spinner("正在分析数据..."):
            try:
                import subprocess
                subprocess.run(["python", "work/audio_review_analysis/main.py", "--all"])
                st.rerun()
            except Exception as e:
                st.error(f"运行分析流程失败: {str(e)}")

# 缓存计算密集型操作
@st.cache_data
def compute_metrics(reviews):
    if not reviews:
        return {}, {}
    
    # 安全计算评分分布
    try:
        # 先将评分转换为浮点数
        ratings = []
        for r in reviews:
            if 'rating' in r and r['rating'] is not None:
                try:
                    ratings.append(float(r['rating']))
                except (ValueError, TypeError):
                    continue
        
        # 然后转为整数进行计数
        rating_counts = pd.Series([int(round(r)) for r in ratings if 1 <= r <= 5]).value_counts()
        rating_dist = {i: rating_counts.get(i, 0) for i in range(1, 6)}
    except Exception as e:
        # 出错时使用默认值
        rating_dist = {i: 0 for i in range(1, 6)}
    
    # 安全计算方面平均分
    aspect_scores = defaultdict(list)
    for review in reviews:
        if 'aspects' in review and review['aspects'] and 'rating' in review:
            try:
                rating = float(review['rating'])
                for aspect in review['aspects']:
                    aspect_scores[aspect].append(rating)
            except (ValueError, TypeError):
                continue
    
    feature_ratings = {
        aspect: np.mean(scores) if scores else 0
        for aspect, scores in aspect_scores.items()
    }
    
    return rating_dist, feature_ratings

# 页面布局优化
def render_header():
    try:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(product_info["image_url"], width=250)
        with col2:
            st.title(product_info["title"])
            st.markdown(f"**价格:** <span style='color: #B12704;'>{product_info['price']}</span>", unsafe_allow_html=True)
            st.markdown(f"**{'⭐' * int(product_info['overall_rating'])}{'☆' if product_info['overall_rating'] % 1 else ''} {product_info['overall_rating']}** ({product_info['total_reviews']} 条评论)")
            st.write("---")
    except Exception as e:
        st.error(f"渲染头部时出错: {str(e)}")

# 优化后的评论分析组件
def render_review_analysis():
    try:
        st.subheader("评论分析概览")
        
        rating_dist, feature_ratings = compute_metrics(all_reviews)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("评分分布", expanded=True):
                total_reviews = sum(rating_dist.values()) or 1  # 避免除以零
                for rating, count in sorted(rating_dist.items(), reverse=True):
                    st.progress(count / total_reviews, text=f"{rating}星 ({count}条, {count/total_reviews:.0%})")
        
        with col2:
            with st.expander("特征评分", expanded=True):
                for feature, rating in feature_ratings.items():
                    st.markdown(f"{feature}: {'⭐' * int(rating)}{'☆' if rating % 1 else ''} {rating:.1f}")
    except Exception as e:
        st.error(f"渲染评论分析时出错: {str(e)}")

# 优化后的评论列表
def render_review_list():
    try:
        st.subheader(f"用户评论 ({len(all_reviews)}条)")
        
        if not all_reviews:
            st.info("没有找到评论数据")
            return
        
        # 添加排序和过滤选项
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("排序方式", 
                                ["最新", "评分最高", "最有帮助", "最相关"])
        with col2:
            filter_rating = st.multiselect("筛选评分", 
                                        [1, 2, 3, 4, 5], 
                                        default=[4, 5])
        
        # 处理排序
        sorted_reviews = all_reviews.copy()
        if sort_by == "最新":
            # 安全处理日期排序
            sorted_reviews.sort(key=lambda x: pd.to_datetime(x.get('date', '2000-01-01'), errors='coerce'), reverse=True)
        elif sort_by == "评分最高":
            # 安全处理评分排序
            sorted_reviews.sort(key=lambda x: float(x.get('rating', 0)) if isinstance(x.get('rating'), (int, float)) or (isinstance(x.get('rating'), str) and x.get('rating').replace('.', '', 1).isdigit()) else 0, reverse=True)
        elif sort_by == "最有帮助":
            # 安全处理帮助票数排序
            sorted_reviews.sort(key=lambda x: float(x.get('helpful_votes', 0)) if isinstance(x.get('helpful_votes'), (int, float)) or (isinstance(x.get('helpful_votes'), str) and x.get('helpful_votes').replace('.', '', 1).isdigit()) else 0, reverse=True)
        
        # 安全处理过滤
        filtered_reviews = []
        for r in sorted_reviews:
            try:
                rating = int(round(float(r.get('rating', 0))))
                if rating in filter_rating:
                    filtered_reviews.append(r)
            except (ValueError, TypeError):
                continue
        
        # 处理空过滤结果
        if not filtered_reviews:
            st.warning("没有符合条件的评论")
            return
        
        # 分页显示
        page_size = st.slider("每页显示", 3, 10, 5)
        max_pages = max(1, (len(filtered_reviews) + page_size - 1) // page_size)
        page = st.number_input("页码", min_value=1, max_value=max_pages, value=1)
        
        start_idx = (page-1)*page_size
        end_idx = min(start_idx + page_size, len(filtered_reviews))
        
        # 在render_review_list函数中，修改评论卡片渲染部分:
        # 渲染评论卡片
        for idx, review in enumerate(filtered_reviews[start_idx:end_idx]):
            with st.container(border=True):
                # ...其他代码保持不变...
                
                # 互动元素
                col1, col2 = st.columns([1,1])
                # 使用idx作为备用ID
                col1.button("👍 有帮助", key=f"helpful_{review.get('id', idx + start_idx)}")
                col2.button("💬 回复", key=f"reply_{review.get('id', idx + start_idx)}")
                
                # 修复评分和标题显示
                rating_colors = ["red", "orange", "yellow", "lime", "green"]
                try:
                    rating = int(round(float(review.get('rating', 3))))
                    rating_idx = max(0, min(rating-1, len(rating_colors)-1))  # 确保索引在有效范围内
                    rating_color = rating_colors[rating_idx]
                except (ValueError, TypeError):
                    rating = 3  # 默认值
                    rating_color = "gray"
                
                st.markdown(
                    f"<span style='color:{rating_color};'>"
                    f"{'⭐'*rating}{'☆'*(5-rating)}</span> "
                    f"<b>{review.get('title', '')}</b>", 
                    unsafe_allow_html=True
                )
                
                # 修复方面标签显示
                if "aspects" in review and review['aspects'] and len(review['aspects']) > 0:
                    aspect_count = len(review['aspects'])
                    if aspect_count > 0:  # 确保有方面可以显示
                        cols = st.columns(aspect_count)
                        for i, (aspect, sentiment) in enumerate(review['aspects'].items()):
                            if i < len(cols):  # 额外安全检查
                                color = "green" if sentiment == "positive" else "red"
                                cols[i].markdown(
                                    f"<span style='color:{color};'>◉ {aspect}</span>", 
                                    unsafe_allow_html=True
                                )
                
                # 评论文本
                st.write(review.get("text", ""))
                
                # 互动元素
                col1, col2 = st.columns([1,1])
                col1.button("👍 有帮助", key=f"helpful_{review.get('id', i)}")
                col2.button("💬 回复", key=f"reply_{review.get('id', i)}")
    except Exception as e:
        st.error(f"渲染评论列表时出错: {str(e)}")

# 主界面
def main():
    try:
        render_header()
        render_review_analysis()
        st.write("---")
        render_review_list()
    except Exception as e:
        st.error(f"应用程序发生错误: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"应用程序启动失败: {str(e)}")
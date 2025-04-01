# dashboard.py (增强版)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt # 仍然需要matplotlib来生成词云图像
import os
import ast # 用于安全地解析字符串化的字典/列表
from collections import Counter
import re

# --- 页面配置 ---
st.set_page_config(
    page_title="音频设备评论分析仪表盘 v2.0",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 样式 (可选) ---
# 可以注入一些简单的CSS来微调外观
st.markdown("""
<style>
    /* 为容器添加更柔和的边框和阴影 */
    .st-emotion-cache- H1 { /* 针对Streamlit特定版本可能需要调整选择器 */
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem; /* 添加一些间距 */
    }
    /* 调整指标卡片 */
    .st-emotion-cache- H1 .stMetric { /* 同样需要检查选择器 */
         background-color: #f8f9fa;
         border-radius: 0.3rem;
         padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --- 数据加载 ---
@st.cache_data
def load_data():
    """加载所有需要的数据集"""
    data_path = 'data/processed/review_aspects.csv'
    aspect_data_path = 'outputs/powerbi_data/aspect_data.csv'
    brand_data_path = 'outputs/powerbi_data/brand_data.csv'
    price_data_path = 'outputs/powerbi_data/price_data.csv'
    aspect_dim_path = 'outputs/powerbi_data/aspect_dim.csv'

    paths = [data_path, aspect_data_path, brand_data_path, price_data_path, aspect_dim_path]
    if not all(os.path.exists(p) for p in paths):
        st.error(f"错误：一个或多个必需的数据文件未找到。请确保路径正确 ({', '.join(paths)}) 并且已运行 `python main.py --all --export-powerbi`。")
        st.stop() # 停止执行

    try:
        df = pd.read_csv(data_path)
        # 尝试安全地转换字符串化的列
        for col in ['aspect_sentences', 'aspect_sentiments']:
            if col in df.columns:
                # 使用fillna('')避免NaN错误，然后应用literal_eval
                 # 注意：如果原始数据就是None或空，literal_eval会失败，需要更复杂的处理
                 # 这里简化处理，假设非空字符串都是有效的字典/列表表示
                df[col] = df[col].fillna('{}').apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('{') else x)

        aspect_data = pd.read_csv(aspect_data_path)
        brand_data = pd.read_csv(brand_data_path)
        price_data = pd.read_csv(price_data_path)
        aspect_dim = pd.read_csv(aspect_dim_path)

        # 处理日期（如果存在）
        if 'review_date' in df.columns:
             df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
             df['review_year_month'] = df['review_date'].dt.to_period('M').astype(str) # 用于分组

        return df, aspect_data, brand_data, price_data, aspect_dim
    except Exception as e:
        st.error(f"加载或处理数据时出错: {e}")
        st.info("请检查CSV文件的格式和内容，特别是 'aspect_sentences' 和 'aspect_sentiments' 列。")
        st.stop()


# --- 全局变量和辅助函数 ---
ASPECT_TRANSLATION = {
    'sound_quality': '音质', 'comfort': '舒适度', 'battery': '电池',
    'connectivity': '连接性', 'noise_cancellation': '降噪', 'build_quality': '做工',
    'controls': '控制', 'price': '价格', 'microphone': '麦克风', 'design': '设计'
}

def get_top_keywords(texts, n=10):
    """从文本列表中提取高频词"""
    words = []
    for text in texts:
        # 简单的分词和清洗
        tokens = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        words.extend(tokens)
    # 可以在这里加入停用词过滤
    return [word for word, count in Counter(words).most_common(n)]

def plot_metric(label, value, help_text=""):
    """自定义指标显示"""
    st.metric(label, value, help=help_text)

def plot_word_cloud(text_series, title):
    """生成并显示词云图"""
    if text_series.empty or text_series.str.strip().eq('').all():
        st.warning(f"{title}：没有足够的文本数据来生成词云。")
        return

    text = ' '.join(text_series.astype(str))
    try:
        wordcloud = WordCloud(
            width=600, height=300, background_color='white',
            colormap='viridis', max_words=150
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except ValueError as e:
         if "negative dimension are not allowed" in str(e):
             st.warning(f"{title}: 文本内容过少或无效，无法生成词云。")
         else:
             st.error(f"生成词云时出错: {e}")


# --- 加载数据 ---
df_main, aspect_data, brand_data, price_data, aspect_dim = load_data()
all_aspects = list(ASPECT_TRANSLATION.keys())
aspect_cols = [f"{a}_score" for a in all_aspects if f"{a}_score" in df_main.columns]
available_aspects = [col.replace('_score', '') for col in aspect_cols]


# --- 侧边栏筛选器 ---
st.sidebar.header("📊 数据筛选")

# 品牌筛选
all_brands = sorted(df_main['brand'].unique().tolist())
# 默认选择评论数最多的前5个品牌
top_brands = df_main['brand'].value_counts().nlargest(5).index.tolist()
selected_brands = st.sidebar.multiselect(
    "选择品牌 (可多选)",
    options=all_brands,
    default=top_brands
)

# 价格区间筛选
all_price_ranges = sorted(df_main['price_range'].unique().tolist())
selected_price_ranges = st.sidebar.multiselect(
    "选择价格区间",
    options=all_price_ranges,
    default=all_price_ranges
)

# 评分范围筛选
min_rating, max_rating = st.sidebar.slider(
    "选择评分范围",
    min_value=1.0, max_value=5.0, value=(1.0, 5.0), step=0.1
)

# 时间范围筛选 (如果日期数据可用)
if 'review_year_month' in df_main.columns:
    all_dates = sorted(df_main['review_year_month'].unique())
    if len(all_dates) > 1:
        selected_date_range = st.sidebar.select_slider(
            "选择评论月份范围",
            options=all_dates,
            value=(all_dates[0], all_dates[-1])
        )
    else:
        selected_date_range = (all_dates[0], all_dates[0]) if all_dates else (None, None)
else:
    selected_date_range = (None, None)


# --- 应用筛选 ---
filtered_df = df_main.copy()
if selected_brands:
    filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
if selected_price_ranges:
    filtered_df = filtered_df[filtered_df['price_range'].isin(selected_price_ranges)]
if min_rating is not None and max_rating is not None:
    filtered_df = filtered_df[(filtered_df['rating'] >= min_rating) & (filtered_df['rating'] <= max_rating)]
if selected_date_range[0] and selected_date_range[1] and 'review_year_month' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['review_year_month'] >= selected_date_range[0]) &
                              (filtered_df['review_year_month'] <= selected_date_range[1])]

# --- 仪表盘主内容 ---
st.title("🎧 音频设备评论分析仪表盘 v2.0")
st.markdown(f"基于 **{len(filtered_df):,}** 条筛选后的评论进行分析。")

if filtered_df.empty:
    st.warning("根据当前筛选条件，没有找到匹配的评论数据。请调整侧边栏的筛选器。")
    st.stop()

# --- 创建标签页 ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 总体概览", "🔍 方面分析", "🏢 品牌比较", "💲 价格分析", "⏱️ 时间趋势", "💬 评论探索"
])

# === Tab 1: 总体概览 ===
with tab1:
    st.header("📈 总体概览")

    with st.container(border=True):
        st.subheader("关键指标")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            plot_metric("评论总数", f"{len(filtered_df):,}")
        with col2:
            avg_r = filtered_df['rating'].mean()
            plot_metric("平均评分", f"{avg_r:.2f} / 5.0" if not pd.isna(avg_r) else "N/A")
        with col3:
            plot_metric("独立品牌数", len(filtered_df['brand'].unique()))
        with col4:
            plot_metric("涉及产品数", len(filtered_df['product_id'].unique()))

    col1, col2 = st.columns([1, 1]) #调整比例

    with col1:
       with st.container(border=True):
            st.subheader("评分分布")
            if not filtered_df.empty:
                rating_counts = filtered_df['rating'].value_counts().sort_index()
                fig_rating = px.bar(
                    x=rating_counts.index, y=rating_counts.values,
                    labels={'x': '评分', 'y': '评论数量'},
                    color=rating_counts.index, color_continuous_scale=px.colors.sequential.Viridis,
                    text_auto=True # 显示数值
                )
                fig_rating.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_rating, use_container_width=True)
            else:
                st.info("无数据显示评分分布。")

    with col2:
        with st.container(border=True):
            st.subheader("评论字数分布")
            if 'word_count' in filtered_df.columns and not filtered_df['word_count'].dropna().empty:
                fig_wc = px.histogram(
                    filtered_df.dropna(subset=['word_count']), x="word_count", nbins=40,
                    labels={'word_count': '评论字数', 'count': '评论数量'},
                    opacity=0.8
                )
                fig_wc.update_layout(bargap=0.1, height=350)
                st.plotly_chart(fig_wc, use_container_width=True)
            else:
                 st.info("无数据显示评论字数分布。")

    with st.container(border=True):
         st.subheader("评论词云")
         col1, col2 = st.columns(2)
         with col1:
             st.markdown("##### 高评分评论 (≥ 4 星)")
             high_rated_text = filtered_df[filtered_df['rating'] >= 4]['clean_review_text']
             plot_word_cloud(high_rated_text, "高评分评论")
         with col2:
             st.markdown("##### 低评分评论 (≤ 2 星)")
             low_rated_text = filtered_df[filtered_df['rating'] <= 2]['clean_review_text']
             plot_word_cloud(low_rated_text, "低评分评论")


# === Tab 2: 方面分析 ===
with tab2:
    st.header("🔍 方面分析")

    col1, col2 = st.columns([1, 1])

    with col1, st.container(border=True):
        st.subheader("各方面平均评分 (雷达图)")
        if not filtered_df.empty and available_aspects:
            aspect_means = {}
            for aspect in available_aspects:
                score = filtered_df[f'{aspect}_score'].mean()
                aspect_means[aspect] = score if not pd.isna(score) else 0

            # 使用翻译后的名称
            categories = [ASPECT_TRANSLATION.get(k, k) for k in aspect_means.keys()]
            values = list(aspect_means.values())

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=categories, fill='toself', name='平均分',
                hovertemplate='<b>%{theta}</b><br>平均分: %{r:.2f}<extra></extra>'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=400, margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("无数据或方面评分列用于生成雷达图。")


    with col2, st.container(border=True):
        st.subheader("方面详细分析")
        selected_aspect = st.selectbox(
            "选择一个方面进行深入分析:",
            options=available_aspects,
            format_func=lambda x: ASPECT_TRANSLATION.get(x, x)
        )

        if selected_aspect:
            score_col = f"{selected_aspect}_score"
            aspect_df = filtered_df.dropna(subset=[score_col])
            if not aspect_df.empty:
                avg_score = aspect_df[score_col].mean()
                median_score = aspect_df[score_col].median()
                st.metric(
                    f"{ASPECT_TRANSLATION.get(selected_aspect, selected_aspect)} 平均分",
                    f"{avg_score:.2f}/10",
                    delta=f"{avg_score - median_score:.2f} vs 中位数 ({median_score:.2f})"
                )

                # 评分分布 (箱线图/小提琴图)
                fig_dist = px.violin(
                    aspect_df, y=score_col, box=True, points="all",
                    labels={score_col: '分数 (1-10)'},
                    title=f'{ASPECT_TRANSLATION.get(selected_aspect, selected_aspect)} 分数分布',
                    hover_data=['rating', 'brand'] # 添加额外信息
                )
                fig_dist.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)

            else:
                st.warning(f"未找到关于 '{ASPECT_TRANSLATION.get(selected_aspect, selected_aspect)}' 的有效评分数据。")


# === Tab 3: 品牌比较 ===
with tab3:
    st.header("🏢 品牌比较")
    min_reviews_brand = st.slider("品牌最少评论数", min_value=5, max_value=100, value=20, key="brand_min_reviews")

    brand_counts = filtered_df['brand'].value_counts()
    valid_brands = brand_counts[brand_counts >= min_reviews_brand].index.tolist()
    top_n_brands = st.number_input("最多显示品牌数", min_value=3, max_value=20, value=10, key="brand_top_n")
    valid_brands = valid_brands[:top_n_brands]

    if not valid_brands:
        st.warning(f"筛选后，没有品牌的评论数达到 {min_reviews_brand} 条。请降低阈值或调整其他筛选器。")
    else:
        st.markdown(f"比较评论数最多的 **{len(valid_brands)}** 个品牌 (评论数 ≥ {min_reviews_brand})。")

        brand_aspect_scores = []
        for brand in valid_brands:
            brand_df = filtered_df[filtered_df['brand'] == brand]
            row = {'Brand': brand, 'Avg Rating': brand_df['rating'].mean(), 'Review Count': len(brand_df)}
            for aspect in available_aspects:
                score = brand_df[f'{aspect}_score'].mean()
                row[ASPECT_TRANSLATION.get(aspect, aspect)] = score if not pd.isna(score) else np.nan
            brand_aspect_scores.append(row)

        brand_scores_pivot = pd.DataFrame(brand_aspect_scores).set_index('Brand')

        with st.container(border=True):
            st.subheader("品牌 vs 方面评分 (热力图)")
            aspect_cols_translated = [ASPECT_TRANSLATION.get(a, a) for a in available_aspects]
            fig_heatmap_brand = px.imshow(
                brand_scores_pivot[aspect_cols_translated].dropna(axis=1, how='all'), # 删除全NaN的列
                labels=dict(x="方面", y="品牌", color="平均分"),
                aspect="auto", color_continuous_scale='Viridis', range_color=[1, 10],
                text_auto=".1f", # 显示一位小数
                title="品牌在各方面的平均表现"
            )
            fig_heatmap_brand.update_layout(height=500 if len(valid_brands) > 5 else 400)
            fig_heatmap_brand.update_xaxes(side="top") # 方面标签放顶部
            st.plotly_chart(fig_heatmap_brand, use_container_width=True)

        with st.container(border=True):
            st.subheader("品牌总体表现")
            brand_summary = brand_scores_pivot[['Avg Rating', 'Review Count']].sort_values('Avg Rating', ascending=False)
            fig_brand_overall = px.bar(
                brand_summary, x=brand_summary.index, y='Avg Rating',
                color='Avg Rating', color_continuous_scale='Viridis', range_color=[1, 5],
                labels={'x': '品牌', 'Avg Rating': '平均总体评分 (1-5)'},
                hover_data={'Review Count': True}, # 悬停显示评论数
                text_auto=".2f"
            )
            fig_brand_overall.update_layout(height=400)
            st.plotly_chart(fig_brand_overall, use_container_width=True)


# === Tab 4: 价格分析 ===
with tab4:
    st.header("💲 价格区间分析")
    min_reviews_price = st.slider("价格区间最少评论数", min_value=5, max_value=100, value=15, key="price_min_reviews")

    price_counts = filtered_df['price_range'].value_counts()
    valid_prices = price_counts[price_counts >= min_reviews_price].index.tolist()
     # 按预定义的顺序排序价格区间
    price_order = ['预算型(<$50)', '中端($50-$150)', '高端($150-$300)', '豪华型(>$300)', '未知']
    valid_prices = sorted(valid_prices, key=lambda x: price_order.index(x) if x in price_order else 99)


    if not valid_prices:
        st.warning(f"筛选后，没有价格区间的评论数达到 {min_reviews_price} 条。请降低阈值或调整其他筛选器。")
    else:
        st.markdown(f"比较 **{len(valid_prices)}** 个价格区间 (评论数 ≥ {min_reviews_price})。")

        price_aspect_scores = []
        for price_range in valid_prices:
            price_df = filtered_df[filtered_df['price_range'] == price_range]
            row = {'Price Range': price_range, 'Avg Rating': price_df['rating'].mean(), 'Review Count': len(price_df)}
            for aspect in available_aspects:
                score = price_df[f'{aspect}_score'].mean()
                row[ASPECT_TRANSLATION.get(aspect, aspect)] = score if not pd.isna(score) else np.nan
            price_aspect_scores.append(row)

        price_scores_pivot = pd.DataFrame(price_aspect_scores).set_index('Price Range')

        with st.container(border=True):
            st.subheader("价格区间 vs 方面评分 (热力图)")
            aspect_cols_translated = [ASPECT_TRANSLATION.get(a, a) for a in available_aspects]
            fig_heatmap_price = px.imshow(
                price_scores_pivot[aspect_cols_translated].dropna(axis=1, how='all'),
                labels=dict(x="方面", y="价格区间", color="平均分"),
                aspect="auto", color_continuous_scale='RdYlGn', range_color=[1, 10], # 红-黄-绿 色阶
                text_auto=".1f", title="不同价格区间在各方面的平均表现"
            )
            fig_heatmap_price.update_layout(height=400)
            fig_heatmap_price.update_xaxes(side="top")
            st.plotly_chart(fig_heatmap_price, use_container_width=True)

        with st.container(border=True):
            st.subheader("方面得分随价格变化趋势")
            price_melted = pd.melt(price_scores_pivot.reset_index(),
                                   id_vars=['Price Range', 'Avg Rating', 'Review Count'],
                                   value_vars=aspect_cols_translated,
                                   var_name='Aspect', value_name='Score')
            fig_price_line = px.line(
                price_melted, x='Price Range', y='Score', color='Aspect',
                markers=True, category_orders={'Price Range': price_order}, # 按顺序绘图
                labels={'Price Range': '价格区间', 'Score': '平均分数 (1-10)', 'Aspect': '方面'},
                title="各方面平均得分在不同价格区间的变化"
            )
            fig_price_line.update_layout(height=450)
            st.plotly_chart(fig_price_line, use_container_width=True)


# === Tab 5: 时间趋势 ===
with tab5:
    st.header("⏱️ 时间趋势分析")
    if 'review_year_month' in filtered_df.columns:
        time_data = filtered_df.dropna(subset=['review_year_month']).copy()
        if not time_data.empty and time_data['review_year_month'].nunique() > 1:

            # 选择分析指标
            time_metric_options = ['平均总体评分'] + [f"平均{ASPECT_TRANSLATION.get(a, a)}得分" for a in available_aspects]
            selected_time_metric = st.selectbox("选择分析指标:", time_metric_options)

            time_agg = None
            y_col = 'Value'
            y_label = selected_time_metric

            if selected_time_metric == '平均总体评分':
                time_agg = time_data.groupby('review_year_month')['rating'].mean().reset_index()
                time_agg.rename(columns={'rating': y_col}, inplace=True)
            else:
                # 从选择的指标中提取原始方面名称
                selected_translated_aspect = selected_time_metric.replace('平均', '').replace('得分', '')
                original_aspect = next((k for k, v in ASPECT_TRANSLATION.items() if v == selected_translated_aspect), None)
                if original_aspect:
                    score_col = f"{original_aspect}_score"
                    if score_col in time_data.columns:
                       time_agg = time_data.groupby('review_year_month')[score_col].mean().reset_index()
                       time_agg.rename(columns={score_col: y_col}, inplace=True)
                       y_label = f"{selected_translated_aspect} 平均分 (1-10)" # 更具体的标签

            if time_agg is not None and not time_agg.empty:
                 with st.container(border=True):
                    st.subheader(f"{selected_time_metric} 随时间变化")
                    fig_time = px.line(
                        time_agg, x='review_year_month', y=y_col, markers=True,
                        labels={'review_year_month': '评论年月', y_col: y_label},
                        title=f"{y_label} 变化趋势"
                    )
                    fig_time.update_layout(height=450)
                    st.plotly_chart(fig_time, use_container_width=True)

                 # 评论量趋势
                 with st.container(border=True):
                     st.subheader("评论数量随时间变化")
                     review_counts_time = time_data['review_year_month'].value_counts().sort_index().reset_index()
                     review_counts_time.columns = ['review_year_month', 'count']
                     fig_count_time = px.bar(
                         review_counts_time, x='review_year_month', y='count',
                         labels={'review_year_month': '评论年月', 'count': '评论数量'},
                         title="每月评论数量"
                     )
                     fig_count_time.update_layout(height=400)
                     st.plotly_chart(fig_count_time, use_container_width=True)

            else:
                st.info("无法计算所选指标的时间趋势。")
        else:
            st.info("时间数据不足 (需要至少两个不同的月份) 或 'review_year_month' 列不存在，无法进行时间趋势分析。")
    else:
         st.warning("评论数据中缺少日期信息 ('review_date' 或 'review_year_month')，无法进行时间趋势分析。")


# === Tab 6: 评论探索 ===
with tab6:
    st.header("💬 评论探索与示例")

    explore_aspect = st.selectbox(
        "选择方面以查看相关评论:",
        options=['(无特定方面)'] + available_aspects,
        format_func=lambda x: ASPECT_TRANSLATION.get(x, x) if x != '(无特定方面)' else x,
        key="explore_aspect"
    )

    sentiment_type = st.radio(
        "评论情感倾向:",
        options=['所有', '正面 (> 6.5分)', '中性 (3.5-6.5分)', '负面 (< 3.5分)'],
        horizontal=True, key="explore_sentiment"
    )

    num_reviews_to_show = st.slider("显示评论数量", 5, 50, 10, key="explore_num")

    # 过滤数据
    explore_df = filtered_df.copy()
    if explore_aspect != '(无特定方面)':
        score_col = f"{explore_aspect}_score"
        explore_df = explore_df.dropna(subset=[score_col]) # 只看有该方面评分的
        if sentiment_type == '正面 (> 6.5分)':
            explore_df = explore_df[explore_df[score_col] > 6.5]
        elif sentiment_type == '中性 (3.5-6.5分)':
            explore_df = explore_df[(explore_df[score_col] >= 3.5) & (explore_df[score_col] <= 6.5)]
        elif sentiment_type == '负面 (< 3.5分)':
            explore_df = explore_df[explore_df[score_col] < 3.5]
    else:
        # 如果没有选特定方面，则根据总体评分过滤
         if sentiment_type == '正面 (> 6.5分)': # 对应 4-5 星
             explore_df = explore_df[explore_df['rating'] >= 4]
         elif sentiment_type == '中性 (3.5-6.5分)': # 对应 3 星
             explore_df = explore_df[explore_df['rating'] == 3]
         elif sentiment_type == '负面 (< 3.5分)': # 对应 1-2 星
             explore_df = explore_df[explore_df['rating'] <= 2]


    if explore_df.empty:
        st.warning("根据当前选择，没有找到匹配的评论。")
    else:
        # 显示相关的关键词 (如果选择了特定方面)
        if explore_aspect != '(无特定方面)' and 'aspect_sentences' in explore_df.columns:
             with st.container(border=True):
                st.subheader(f"关于 '{ASPECT_TRANSLATION.get(explore_aspect, explore_aspect)}' 的常见词")
                try:
                     # 提取与选定方面相关的句子文本
                     relevant_sentences = []
                     for sentences_dict in explore_df['aspect_sentences']:
                         # 检查是否是有效字典以及是否包含选定的方面
                         if isinstance(sentences_dict, dict) and explore_aspect in sentences_dict:
                              relevant_sentences.extend(sentences_dict[explore_aspect])

                     if relevant_sentences:
                         keywords = get_top_keywords(relevant_sentences, n=15)
                         st.info(f"**常见词:** {', '.join(keywords)}")
                     else:
                         st.info("未能提取相关句子或关键词。")
                except Exception as e:
                     st.error(f"提取关键词时出错: {e}")
                     st.info("可能的原因：'aspect_sentences'列的格式不正确或为空。")


        # 显示示例评论
        with st.container(border=True):
            st.subheader("示例评论")
            # 优先显示有文本的评论，并按某种方式排序（例如，按评分极端性）
            display_df = explore_df.dropna(subset=['clean_review_text']).sample(min(num_reviews_to_show, len(explore_df)))

            for _, row in display_df.iterrows():
                 with st.expander(f"**{row.get('brand', 'N/A')}** - {row.get('product_id', 'N/A')} (评分: {row.get('rating', 'N/A')})"):
                    st.markdown(f"**标题:** {row.get('clean_review_title', '(无标题)')}")
                    st.markdown(f"**评论:**")
                    st.markdown(f"> {row.get('clean_review_text', '(无内容)')}")
                    # 显示该评论的方面得分 (如果存在)
                    aspect_scores_in_review = {}
                    for aspect in available_aspects:
                         score_col = f"{aspect}_score"
                         if score_col in row and not pd.isna(row[score_col]):
                             aspect_scores_in_review[ASPECT_TRANSLATION.get(aspect, aspect)] = f"{row[score_col]:.1f}"
                    if aspect_scores_in_review:
                        st.markdown("**方面评分:** " + " | ".join([f"{k}: {v}" for k,v in aspect_scores_in_review.items()]))


# --- 页脚 ---
st.markdown("---")
st.markdown("© 2025 音频设备评论分析系统 | 现代仪表盘 Demo")
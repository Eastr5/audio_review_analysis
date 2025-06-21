import pandas as pd
import os
from src.visualization import plots 

# --- 配置 ---
PROCESSED_DATA_PATH = 'data/processed/review_aspects.csv'
OUTPUT_HTML_DIR = 'outputs/html_dashboard'
MAIN_HTML_PATH = os.path.join(OUTPUT_HTML_DIR, 'dashboard.html')

def generate_dashboard():
    """主函数，生成所有图表并创建HTML仪表盘"""
    
    # --- 确保输出目录存在 ---
    os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)
    print(f"输出目录 '{OUTPUT_HTML_DIR}' 已准备好。")

    # --- 加载数据 ---
    print("正在加载已处理的方面分析数据...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print("数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 未找到方面分析数据文件 '{PROCESSED_DATA_PATH}'。")
        print("请先成功运行 `python main.py --all` 来生成该文件。")
        return

    # --- 生成并保存所有独立的交互式图表 ---
    plot_functions = {
        'rating_distribution': plots.plot_rating_distribution,
        'aspect_scores_radar': plots.plot_aspect_scores,
        'price_range_comparison': plots.plot_price_range_comparison,
        'brand_comparison_heatmap': plots.plot_brand_comparison,
    }

    plot_paths = {}
    for name, func in plot_functions.items():
        print(f"--- 正在生成 {name} 图表 ---")
        fig = func(df, OUTPUT_HTML_DIR)
        if fig:
            path = os.path.join(OUTPUT_HTML_DIR, f'{name}.html')
            fig.write_html(path, full_html=False, include_plotlyjs='cdn')
            plot_paths[name] = f'{name}.html' # 相对路径
            print(f"图表已保存到: {path}")
        else:
            print(f"跳过 {name} 图表生成（函数未返回图表对象）。")

    # --- 创建主 Dashboard HTML 文件 ---
    print("\n--- 正在创建主仪表盘 HTML 文件 ---")
    
    # 动态生成 iframe 卡片
    cards_html = ""
    plot_titles = {
        'rating_distribution': "整体评分分布",
        'aspect_scores_radar': "音频设备各方面平均分",
        'price_range_comparison': "不同价格区间的方面评分",
        'brand_comparison_heatmap': "品牌方面评分比较热力图",
    }
    
    for name, path in plot_paths.items():
        title = plot_titles.get(name, "图表")
        cards_html += f"""
        <div class="card">
            <h2>{title}</h2>
            <iframe src="{path}"></iframe>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>音频产品评论分析仪表盘</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f2f6; color: #333; }}
            .container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; max-width: 1800px; margin: auto; }}
            .card {{ background-color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; overflow: hidden;}}
            h1, h2 {{ text-align: center; color: #1e3a8a; }}
            iframe {{ width: 100%; height: 550px; border: none; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <h1>音频产品评论分析仪表盘 (静态版)</h1>
        <div class="container">
            {cards_html}
        </div>
        <div class="footer">
            <p>这是一个静态生成的报告。图表可交互（悬停、缩放），但无法进行全局筛选。</p>
            <p>报告生成于: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

    with open(MAIN_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n主仪表盘已成功创建！请在浏览器中打开文件: {MAIN_HTML_PATH}")

if __name__ == "__main__":
    generate_dashboard()
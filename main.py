import os
import pandas as pd
from src.data.acquisition import get_audio_dataset
from src.data.preprocessing import get_processed_dataset
from src.features.aspect_extraction import create_review_aspects_dataset
from src.visualization.plots import create_all_visualizations
import argparse

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
    
    # 5. 导出FineBi数据
    if args.export_finebi:
        print("\n=== 步骤 5: 导出FineBi数据 ===")
        # 导入新的导出模块
        from src.export.excel_export import export_to_excel_for_finebi
        
        # 如果没有加载数据，尝试从文件加载
        if aspect_data is None:
            try:
                aspect_data = pd.read_csv('data/processed/review_aspects.csv')
                print("从文件加载方面分析数据。")
            except FileNotFoundError:
                print("未找到方面分析数据文件。请先运行 --aspect-analysis 步骤。")
                return
        
        # 调用导出函数
        output_path = 'outputs/finebi_data.xlsx'
        export_to_excel_for_finebi(aspect_data, output_path)
        print(f"已将数据导出至 {output_path} 用于FineBi仪表盘")
    
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
    parser.add_argument('--export-finebi', action='store_true', 
                        help='导出数据用于FineBi仪表盘')
    parser.add_argument('--data-url', type=str, 
                        default="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
                        help='数据集URL')
    
    args = parser.parse_args()
    
    # 如果没有指定任何步骤，默认运行所有步骤
    if not any([args.all, args.data_collection, args.preprocessing, 
                args.aspect_analysis, args.visualization, 
                args.export_powerbi, args.export_finebi]):
        args.all = True
    
    main(args)

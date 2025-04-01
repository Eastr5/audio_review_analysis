# 音频设备评论分析系统 - 使用手册

## 1. 系统安装

### 1.1 环境要求
- Python 3.8+
- pip 20.0+
- 推荐使用虚拟环境

### 1.2 安装步骤
```bash
# 克隆项目仓库
git clone [项目地址]

# 进入项目目录
cd audio_review_analysis

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 2. 配置指南

### 2.1 配置文件说明 (config.py)
```python
# 数据源配置
DATA_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"

# 音频设备关键词
AUDIO_KEYWORDS = [
    "headphone", "earphone", "earbud", 
    "headset", "speaker", "audio"
]

# 分析方面配置
ASPECTS = {
    "音质": ["sound", "audio", "bass", "treble"],
    "舒适度": ["comfort", "fit", "ear"],
    # ...其他方面配置
}
```

### 2.2 自定义配置
1. 修改数据源URL
2. 添加/删除音频关键词
3. 调整分析方面

## 3. 使用教程

### 3.1 完整分析流程
```bash
python main.py --all
```
流程包括：
1. 数据获取
2. 数据预处理
3. 方面分析
4. 可视化生成
5. 数据导出

### 3.2 分步执行
```bash
# 仅获取数据
python main.py --data-collection

# 预处理数据
python main.py --preprocessing

# 方面分析
python main.py --aspect-analysis

# 生成可视化
python main.py --visualization

# 导出PowerBI数据
python main.py --export-powerbi
```

## 4. 数据分析仪表盘

### 4.1 启动仪表盘
```bash
streamlit run consumer_dashboard.py
```

### 4.2 功能说明
- 产品评分概览
- 方面评分对比
- 评论关键词云
- 品牌比较
- 评论筛选功能

## 5. 高级功能

### 5.1 自定义分析
```python
from src.data.acquisition import get_audio_dataset
from src.features.aspect_extraction import analyze_aspects

# 获取自定义数据
data = get_audio_dataset(custom_url="your_data_url")

# 自定义方面分析
results = analyze_aspects(
    data,
    aspects={
        "新方面": ["keyword1", "keyword2"]
    }
)
```

### 5.2 扩展可视化
```python
from src.visualization.plots import create_custom_plot

create_custom_plot(
    data,
    plot_type="radar",
    output_file="custom_plot.html"
)
```

## 6. 常见问题解答

### 6.1 数据获取失败
- 检查网络连接
- 验证数据源URL
- 确认有足够存储空间

### 6.2 分析结果不准确
- 检查关键词配置
- 调整情感分析阈值
- 验证数据预处理效果

### 6.3 可视化问题
- 检查matplotlib版本
- 确认输出目录权限
- 验证数据格式

## 7. 最佳实践

1. **定期更新关键词**：根据新产品特性更新分析关键词
2. **数据备份**：定期备份raw和processed数据
3. **结果验证**：人工抽样检查分析结果
4. **性能优化**：大数据集时使用分块处理

## 附录

### A. 命令行参数参考
| 参数 | 说明 |
|------|------|
| --all | 执行完整流程 |
| --data-collection | 仅数据获取 |
| --aspect-analysis | 仅方面分析 |
| --export-powerbi | 导出PowerBI数据 |

### B. 文件结构说明
[同前文技术架构部分]

### C. 联系支持
- 邮箱: support@audio-analysis.com
- 文档: https://docs.audio-analysis.com

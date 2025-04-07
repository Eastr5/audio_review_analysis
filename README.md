# README.md

# 🎧 音频产品评论分析与可视化平台

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/libraries-pandas%7Cplotly%7Cstreamlit-orange.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
## 📝 项目简介

本项目旨在分析音频产品（如耳机、音响）的用户在线评论，自动提取用户关注的关键产品方面（如音质、电池续航、舒适度等），分析用户对这些方面的情感倾向，并通过交互式仪表板和可视化图表展示分析结果。旨在为产品改进、市场分析和消费者决策提供数据支持。

## ✨ 核心功能

* **数据预处理:** 清洗和规范化原始评论文本数据。
* **方面提取 (ABSA):** 基于关键词识别评论中提及的产品方面（例如 "音质", "电池", "舒适度"）。
* **情感关联:** 将评论的总体评分作为对应方面的情感代理。
* **多维度可视化:** 使用 Plotly 生成交互式图表，包括：
    * 整体评分分布
    * 品牌/产品在各方面的平均得分（雷达图、热力图）
    * 不同价格区间的产品表现对比
* **交互式仪表板:** 使用 Streamlit 构建用户友好的 Web 应用，允许用户筛选品牌、方面、价格范围等，动态查看分析结果。
* **BI 数据导出:** 输出适用于 Power BI / FineBI 等商业智能工具的数据模型。

## 🚀 技术栈

* **核心语言:** Python 3.8+
* **数据处理:** Pandas, NumPy
* **数据可视化:** Plotly
* **Web 仪表板:** Streamlit
* **进度条:** Tqdm (用于主流程)
* **配置管理:** Python 文件 (`config.py`)
* **版本控制:** Git / GitHub 

## 📂 项目结构

audio_review_analysis/<br>
├── config.py               # 配置文件 (路径, 关键词等)<br>
├── main.py                 # 主分析流程入口<br>
├── dashboard.py            # Streamlit 分析师仪表板应用<br>
├── consumer_dashboard.py   # Streamlit 消费者仪表板应用 (可选)<br>
├── requirements.txt        # Python 依赖库<br>
├── README.md               # 项目说明 (本文档)<br>
├── DETAILED_DOCS.md        # 详细技术文档<br>
├── USER_GUIDE.md           # 用户操作指南<br>
├── OPTIMIZATION.md         # 优化相关说明<br>
├── data/<br>
│   ├── raw/                # 存放原始数据 (需要自行准备或使用提供的样本)<br>
│   │   └── sample_audio_reviews_raw.csv (示例样本数据)<br>
│   └── processed/          # 存放处理后的数据 (由 main.py 生成)<br>
│       ├── audio_reviews.csv<br>
│       └── review_aspects.csv<br>
├── outputs/<br>
│   ├── figures/            # 存放生成的 HTML 图表 (由 main.py 生成)<br>
│   ├── powerbi_data/       # 存放为 Power BI 导出的数据 (由 main.py 生成)<br>
│   └── finebi_data.xlsx - 主数据.csv # 为 FineBI 导出的数据 (由 main.py 生成)<br>
└── src/                    # 源代码目录<br>
├── data/               # 数据处理模块 (acquisition.py, preprocessing.py)<br>
├── features/           # 特征工程/核心分析模块 (aspect_extraction.py)<br>
├── visualization/      # 可视化模块 (plots.py)<br>
└── export/             # 数据导出模块 (powerbi_export.py)<br>


## ⚙️ 安装

1.  **克隆仓库:**
    ```bash
    cd audio_review_analysis
    ```
2.  **创建并激活虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 使用说明

1.  **准备数据:**
    * **使用样本数据 (推荐首次运行):** 项目在 `data/raw/` 目录下提供了一个小型的匿名样本文件 `sample_audio_reviews_raw.csv`。`config.py` 默认可能指向这个样本文件，或者你需要修改 `config.py` 中的 `RAW_REVIEWS_FILE` 指向它。
    * **使用你自己的数据:** 将你的原始评论 CSV 文件放入 `data/raw/` 目录，并确保其列名与项目预期一致（至少包含 `review_id`, `review_text`, `rating`, `brand`, `price`）。然后修改 `config.py` 中的 `RAW_REVIEWS_FILE` 指向你的文件名。

2.  **运行分析流程:**
    * 在项目根目录下运行 `main.py` 脚本。这将执行数据预处理、方面提取、可视化图表生成和 BI 数据导出。
    * ```bash
        python main.py
        ```
    * 处理后的数据将保存在 `data/processed/`，图表和导出数据将保存在 `outputs/`。

3.  **启动仪表板:**
    * 运行 Streamlit 命令启动主仪表板：
    * ```bash
        streamlit run dashboard.py
        ```
    * (可选) 启动消费者版本仪表板：
    * ```bash
        streamlit run consumer_dashboard.py
        ```
    * 在浏览器中打开显示的本地 URL (通常是 `http://localhost:8501`) 即可查看和交互。

## 📊 仪表板演示



## 🤝 贡献

欢迎提出改进意见或贡献代码！请通过 GitHub Issues 或 Pull Requests 进行。

## 📧 联系

* [张世昌] - [2022312106@eamil.cufe.edu.cn] - [https://github.com/Eastr5]

## 📄 许可

本项目采用 [MIT License](LICENSE) 授权。 ```




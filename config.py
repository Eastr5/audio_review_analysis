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

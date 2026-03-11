import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

# 下载官方教程数据集（很小，几秒钟）
dataset = load_dataset(
    "huggingface-course/vqa",  # 修正后的名字
    cache_dir="./vqa_tutorial"
)
print("下载成功！数据集结构：")
print(dataset)
print(dataset["train"][0])  # 看一条数据长什么样
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    model_id='qwen/Qwen2.5-VL-3B-Instruct',  # 改为 3B
    cache_dir='./Qwen2.5-VL-3B-Instruct',
    revision='master'
)
print("模型下载完成！")
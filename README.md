# Qwen2.5-VL 医学视觉问答微调

本项目基于 Qwen2.5-VL-3B 模型，在 SimpleVQA 医学图像问答数据集上进行 LoRA 微调，实现医学影像理解与诊断辅助问答。

## 🚀 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.5+
- transformers 5.3.0
- peft, bitsandbytes, qwen-vl-utils

安装依赖：
```bash
pip install -r requirements.txt
```

### 数据准备
下载 SimpleVQA 数据集（已包含在项目中），或自行准备 parquet 格式的数据，需包含 `image`, `question`, `answer` 字段。

### 训练
```bash
python train_vqa.py
```

训练参数可在脚本中调整，训练日志默认保存至 `./qwen2vl_vqa_lora/`，最终 LoRA 权重保存至 `./qwen2vl_vqa_lora_final/`。

### 推理
```bash
python inference.py
```
默认读取 `test.jpg` 作为输入图像，可根据需要修改图像路径和问题文本。

## 📊 训练结果
- 训练集大小：2025 条
- 验证集大小：2025 条
- 训练时间：约 59 分钟（3 epoch）
- 最终训练损失：**7.04**
- 梯度范数：0.06267（稳定收敛）

### 损失下降曲线（示例）
| Step | Loss | Gradient Norm |
|------|------|---------------|
| 0    | 17.31| 0.805         |
| 72   | 7.695 | 0.7107        |
| 762  | 7.04  | 0.06267       |

### 推理示例
输入图像（从 SimpleVQA 测试集中抽取）：
```
[图像]
```
用户提问："箭头所指的是什么穴位。"
模型回答："伏兔穴"

## 局限性及未来工作

尽管模型在 SimpleVQA 测试集上取得了较好的收敛结果（训练损失降至 7.04），但在实际推理中发现了一些的局限性：

- **穴位识别不稳定**：在对同一穴位图像进行多次测试时，模型偶尔会输出不同的答案，例如在指向“伏兔穴”的图像上，有时会误判为“环跳穴”。这表明模型对局部解剖特征的分辨能力尚有不足，可能受训练数据中穴位类别样本不平衡或特征相似性影响。
- **对视觉提示敏感**：当我添加箭头以引导模型关注特定区域时，模型的输出有时会发生偏移，说明空间注意力机制不够鲁棒。

未来我们将尝试以下改进：
1. 收集更多穴位图像，平衡各类别样本；
2. 引入目标检测模块，先定位关键点再进行分类；
3. 尝试多视角数据增强，提升模型对空间位置的鲁棒性。

## 🛠️ 项目结构
```
.
├── train_vqa.py          # 训练脚本
├── inference.py          # 推理脚本
├── requirements.txt      # 依赖列表
├── .gitignore            # Git 忽略文件
├── SimpleVQA/            # 数据集目录
│   ├── test.parquet
│   └── train.parquet
└── README.md             # 本文档
```

## 📌 注意事项
- 模型文件较大（3B 约 6GB），未包含在仓库中，请自行下载并放置于 `./Qwen2.5-VL-3B-Instruct/qwen/Qwen2___5-VL-3B-Instruct`。
- 如遇网络问题，可使用镜像站或手动下载。

## 📄 License
MIT

## 👤 作者
Chen Yang

## 🙏 致谢
- [Qwen2.5-VL 官方仓库](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [SimpleVQA 数据集](https://huggingface.co/datasets/m-a-p/SimpleVQA)

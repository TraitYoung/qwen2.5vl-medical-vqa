import os
import base64
import io
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

# -------------------- 数据集类（返回原始数据）--------------------
class VQADataset(Dataset):
    def __init__(self, parquet_path, data_limit=None):
        self.df = pd.read_parquet(parquet_path)
        if data_limit:
            self.df = self.df.head(data_limit)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # 解码图像（base64 -> PIL Image）
        image_base64 = item['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        try:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"图像解码失败，索引 {idx}，错误：{e}")
            image = Image.new('RGB', (224, 224), color='gray')

        return {
            "image": image,
            "question": item['question'],
            "answer": item['answer']
        }

# -------------------- 自定义 collate_fn（关键修复）--------------------
def collate_fn(batch, processor):
    # 构建对话格式
    messages = []
    for item in batch:
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["question"]}
            ]},
            {"role": "assistant", "content": item["answer"]}
        ]
        messages.append(conversation)

    # 提取图像信息
    images = [process_vision_info(msg)[0] for msg in messages]

    # 应用 chat template
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
             for msg in messages]

    # 一次性处理整个batch
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
        min_pixels=224 * 224,
        max_pixels=1280 * 28 * 28
    )

    # 🚀 关键修复：构造 labels，让模型计算损失
    labels = inputs["input_ids"].clone()
    # 将 padding 部分设为 -100（忽略损失）
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels

    return inputs

# -------------------- 主训练 --------------------
def main():
    # 模型路径
    model_id = "./Qwen2.5-VL-3B-Instruct/qwen/Qwen2___5-VL-3B-Instruct"

    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )

    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 数据集路径
    train_path = "./SimpleVQA/train.parquet"
    val_path = "./SimpleVQA/test.parquet"

    if not os.path.exists(train_path):
        print(f"警告：{train_path} 不存在，使用 {val_path} 作为训练集")
        train_path = val_path

    train_dataset = VQADataset(train_path)
    val_dataset = VQADataset(val_path)

    print(f"训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./qwen2vl_vqa_lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_8bit",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    # 绑定 processor 到 collate_fn
    def data_collator_fn(batch):
        return collate_fn(batch, processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator_fn,
    )

    trainer.train()
    trainer.save_model("./qwen2vl_vqa_lora_final")
    processor.save_pretrained("./qwen2vl_vqa_lora_final")

if __name__ == "__main__":
    main()
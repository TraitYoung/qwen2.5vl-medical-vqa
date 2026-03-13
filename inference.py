import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# 加载基础模型
base_model_path = "./Qwen2.5-VL-3B-Instruct/qwen/Qwen2___5-VL-3B-Instruct"
lora_path = "./qwen2vl_vqa_lora_final"

print("加载基础模型...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("加载LoRA权重...")
model = PeftModel.from_pretrained(base_model, lora_path)

# 加载处理器
processor = AutoProcessor.from_pretrained(lora_path, trust_remote_code=True)

# 测试图像（请替换为你的医学图像路径）
image_path = "ct-images-in-covid19.png"  # 这里放一张测试图像
image = Image.open(image_path).convert("RGB")

# 构造对话
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "这是什么病导致的肺部图像。"}
        ]
    }
]

# 应用对话模板并生成
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

# 生成回答
output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.1
)
output_text = processor.decode(output_ids[0], skip_special_tokens=True)

print("\n=== 模型回答 ===")
print(output_text)
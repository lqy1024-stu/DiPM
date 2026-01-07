import os
import json
import csv
import torch
import logging
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------- CSV ----------------
def init_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "question", "answer"])

# ---------------- Data ----------------
def load_party_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------- Model ----------------
model_path = ""
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

lora_path = ''
model = PeftModel.from_pretrained(model, model_id=lora_path)

# ---------------- Inference ----------------
def evaluate_sample(instruction):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    outputs = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- Main Loop ----------------
def read_all_questions_one_by_one(data, csv_path):
    init_csv(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx, row in enumerate(data):
            print(row)
            print(f"问题 {idx}: {row['issue']}\n")
            answer = evaluate_sample(row['issue'])
            print(f"答案 {idx}: {answer}\n")
            writer.writerow([
                idx,
                row['issue'],
                answer
            ])
            f.flush()

# ---------------- Run ----------------
file_path = '' # test data
csv_output_path = 'dataset/qwen_lora_shifted_5.csv' # 用于后续的政治立场评分
data = load_party_questions(file_path)
read_all_questions_one_by_one(data, csv_output_path)

import pandas as pd  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import logging
from peft import PeftModel  # type: ignore
import json
import random
import os

# 固定随机种子保证可复现
random.seed(42)

# ======================
# 模型与 LoRA 配置
# ======================
model_path = ""
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 以下是lora模块的路径
# lora_path = ''

model = PeftModel.from_pretrained(model, model_id=lora_path)

def generate_prompt(instruction):
    return f"{instruction}"

def evaluate_sample(instruction, gold_label):
    prompt = generate_prompt(instruction)
    log_message(f"prompt: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        use_cache=True
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].lower()

    log_message(f"response: {response}")
    return response in gold_label.lower(), response


# ======================
# 多数据集循环评估
# ======================
test_sets = {
    "mrpc": "dataset/test_set/mrpc_test.json",
    "cola": "dataset/test_set/cola_test.json",
    "mnli": "dataset/test_set/mnli_test.json",
    "rte":  "dataset/test_set/rte_test.json",
}

results = {}

for task_name, test_path in test_sets.items():
    print(f"\n===== Evaluating {task_name.upper()} =====")

    # 读取数据
    with open(test_path, "r", encoding="utf-8") as f:
        sampled_data = json.load(f)

    correct = 0
    total = len(sampled_data)

    for idx, row in enumerate(sampled_data):
        is_correct, predicted = evaluate_sample(
            row["instruction"],
            row["output"]
        )

        if is_correct:
            correct += 1

    acc = correct / total
    results[task_name] = acc

    print(f"{task_name.upper()} accuracy = {acc:.4%}")

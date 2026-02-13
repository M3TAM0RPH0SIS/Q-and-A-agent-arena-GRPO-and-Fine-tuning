import torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainingArguments
from datasets import load_dataset
from rewards import format_reward_func, complexity_reward_func
import os

# --- CONFIG ---
os.environ["WANDB_DISABLED"] = "true"
MODEL_PATH = "q_agent_lora_local" 
OUTPUT_DIR = "q_agent_grpo_final"

# 1. Load SFT Model
print(f"🚀 Loading SFT Model from {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 2048,
    load_in_4bit = True,
    gpu_memory_utilization = 0.6,
)

# 2. Prepare Data (THE FIX IS HERE)
# We load your questions, but we must add a "prompt" column.
# The prompt is the "trigger" that tells the model to start generating.
dataset = load_dataset("json", data_files="hard_questions.jsonl", split="train")

# This template MUST match what you used in train_sft.py
system_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert examiner. Generate a difficult multiple-choice question about Python Memory Management.

### Input:
Generate a hard question.

### Response:
"""

def add_prompt_column(example):
    return {
        # This is the KEY that GRPOTrainer looks for
        "prompt": system_prompt
    }

# Apply this to every row in your dataset
dataset = dataset.map(add_prompt_column)

# 3. Config 
training_args = GRPOConfig(
    output_dir = OUTPUT_DIR,
    learning_rate = 5e-6, 
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    per_device_train_batch_size = 1, # Strict 1 for 8GB VRAM
    gradient_accumulation_steps = 4,
    num_generations = 4, 
    max_steps = 30, # Short run
    report_to = "none",
    use_vllm = False, 
)

# 4. Train
print("⚔️ Starting GRPO Training...")
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [format_reward_func, complexity_reward_func],
    args = training_args,
    train_dataset = dataset,
    # Pass length arguments here (fix for the TypeError)
    max_prompt_length = 256,
    max_completion_length = 512,
    processing_class = tokenizer,
)

trainer.train()

# 5. Save
print("💾 Saving RL Agent...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ GRPO FINISHED!")
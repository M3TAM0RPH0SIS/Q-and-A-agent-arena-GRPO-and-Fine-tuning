import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# --- CRITICAL WINDOWS FIXES ---
os.environ["WANDB_DISABLED"] = "true" # Disable WandB for now (avoid login prompts)

# 1. Config
# We use the 1.5B model. It is small but smart enough to learn the FORMAT.
# At the hackathon, you just change this string to "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "q_agent_checkpoint"
MAX_SEQ_LENGTH = 2048 

# 2. Load Model
print(f"🚀 Loading {MODEL_NAME} on RTX 5050...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto-detects Bfloat16 (supported on your GPU!)
    load_in_4bit = True, # Mandatory for 8GB VRAM
)

# 3. Add LoRA Adapters (The "Fine-Tuning" layers)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 4. Format Data
print("📂 Preparing Dataset...")
dataset = load_dataset("json", data_files="hard_questions.jsonl", split="train")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert examiner. Generate a difficult multiple-choice question about Python Memory Management.

### Input:
{input_context}

### Response:
{output_json}"""

def formatting_prompts_func(examples):
    texts = []
    # Handle cases where your JSON might have slightly different keys
    questions = examples.get("question", [])
    options = examples.get("options", [])
    answers = examples.get("answer", [])
    explanations = examples.get("explanation", [])
    
    for q, opts, ans, exp in zip(questions, options, answers, explanations):
        json_output = f'{{"question": "{q}", "options": {opts}, "answer": "{ans}", "explanation": "{exp}"}}'
        text = alpaca_prompt.format(
            input_context="Generate a hard question.", 
            output_json=json_output
        ) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Train
print("⚔️ Starting Local Training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 1, # Windows doesn't like multi-process data loading
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Safe for 8GB
        gradient_accumulation_steps = 4, # Simulates batch size 8
        warmup_steps = 5,
        max_steps = 60, # Quick run to prove it works
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = OUTPUT_DIR,
        report_to = "none", # Don't try to upload stats
    ),
)

trainer.train()

# 6. Save
print("💾 Saving Adapter...")
model.save_pretrained("q_agent_lora_local")
tokenizer.save_pretrained("q_agent_lora_local")
print("✅ DONE! You are ready for the Hackathon.")
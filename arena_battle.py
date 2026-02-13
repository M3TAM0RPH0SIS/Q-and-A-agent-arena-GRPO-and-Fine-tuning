import torch
from unsloth import FastLanguageModel
import gc
import json
from a_agent import AnswerAgent

# --- ROUND 1: ATTACK (Q-Agent) ---
print("\n🔥 ROUND 1: Q-Agent Generating...")

# 1. Load the SFT Model (Let's verify the SFT first)
# If this works, we know the model is good, and GRPO just needs tuning.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "q_agent_grpo_final", # Checkpoint from Stage 2
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. The EXACT Prompt used in training (Crucial!)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert examiner. Generate a difficult multiple-choice question about Python Memory Management.

### Input:
Generate a hard question.

### Response:
"""

# 3. Generate
inputs = tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs, 
    max_new_tokens=512, 
    temperature=0.7 # Add a little creativity
)

# 4. Decode
q_output = tokenizer.batch_decode(outputs)[0]
# Extract just the JSON part after "Response:"
q_json_str = q_output.split("### Response:")[-1].strip()

# Clean up any trailing tokens (sometimes it adds <|endoftext|>)
if "<|im_end|>" in q_json_str:
    q_json_str = q_json_str.split("<|im_end|>")[0]

print(f"Generated Question:\n{q_json_str}")

# CLEANUP VRAM
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# --- ROUND 2: DEFENSE (A-Agent) ---
print("\n🛡️ ROUND 2: A-Agent Attempting to Solve...")
# Load A-Agent (Fresh)
defender = AnswerAgent(model_name="Qwen/Qwen2.5-1.5B-Instruct")
my_answer = defender.solve(q_json_str)

print(f"\n🏆 Final Result:")
print(f"A-Agent Answered: {my_answer}")
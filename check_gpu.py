import os
import torch

# --- CRITICAL FIXES FOR WINDOWS ---
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCH_COMPILE_DEBUG"] = "0"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from unsloth import FastLanguageModel

print(f"Pytorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "Qwen/Qwen2.5-0.5B-Instruct",
            max_seq_length = 512,
            load_in_4bit = True,
        )
        print("✅ SUCCESS: Unsloth loaded the model!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
else:
    print("❌ ERROR: CUDA not detected.")
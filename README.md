# Q-and-A-agent-arena-GRPO-and-Fine-tuning

---

#  The Q-Agent vs. A-Agent Arena

### **Reinforcement Learning (GRPO) for Adversarial LLM Testing**

This repository contains a full pipeline for training and evaluating **Adversarial LLMs** in a competitive "Question-Answering" battle. The project utilizes **Supervised Fine-Tuning (SFT)** followed by **Group Relative Policy Optimization (GRPO)** to evolve a model from a general assistant into a specialized, "nasty" examiner.

---

## The Architecture

Our pipeline follows a three-stage evolutionary process:

1. **Stage 1: Seed Generation:** Generating high-reasoning "Golden Data" in JSONL format using high-parameter models (Groq/Llama-3).
2. **Stage 2: Supervised Fine-Tuning (SFT):** Training a proxy model (Qwen-1.5B/14B) via **Unsloth** to master the strict JSON schema and domain-specific knowledge (Python Memory Management).
3. **Stage 3: Reinforcement Learning (GRPO):** Optimizing the model's strategy using a reward-based system. We reward **Format Accuracy**, **Logical Complexity**, and **Solvability Gaps**.

---

##  File Structure & Workflow

| File | Purpose |
| --- | --- |
| `groq_generate.py` | Generates the initial 100+ "hard" question seeds. |
| `hard_questions.jsonl` | The "Golden Dataset" used for initial training. |
| `train_sft.py` | Fine-tunes the base model to understand the examination format. |
| `rewards.py` | **The Brain.** Defines the reward functions for GRPO (Format & Complexity). |
| `train_grpo.py` | The RL trainer. Uses GRPO to maximize the "nastiness" of questions. |
| `a_agent.py` | The Defender. A Chain-of-Thought (CoT) agent designed to solve questions. |
| `arena_battle.py` | The testing ground where the Q-Agent attacks the A-Agent. |
| `check_gpu.py` | Sanity check for CUDA, PyTorch versions, and Unsloth compatibility. |

---

##  How to Run

### **1. Environment Setup**

Ensure you are using **CUDA 12.1+** and the specific versions of Unsloth/Trl that support GRPO.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

```

### **2. The Training Loop**

Run the SFT stage first to teach the model the "rules":

```powershell
python train_sft.py

```

Then, run the GRPO stage to teach the model "strategy":

```powershell
python train_grpo.py

```

### **3. The Battle**

Evaluate the performance of your Attacker against the Defender:

```powershell
python arena_battle.py

```

---

##  Hackathon Strategy

This agent is specifically tuned to exploit **Python Memory Management** quirks (Circular references, `__del__` finalizers, WeakRef traps). By utilizing **GRPO**, our model learns to generate questions that are logically sound yet statistically difficult for standard LLMs to parse without deep reasoning.


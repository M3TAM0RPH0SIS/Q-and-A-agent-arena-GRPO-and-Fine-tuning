import torch
from unsloth import FastLanguageModel
import json
import re

class AnswerAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"🛡️ Loading A-Agent ({model_name})...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 2048,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model)

    def solve(self, question_json):
        """
        Takes a JSON question and returns the answer (A/B/C/D).
        """
        # Parse the input
        try:
            if isinstance(question_json, str):
                data = json.loads(question_json)
            else:
                data = question_json
                
            q_text = data.get("question", "")
            options = data.get("options", [])
        except:
            return "ERROR"

        # The "Reasoning" Prompt
        prompt = f"""You are a world-class Python expert.
Question: {q_text}
Options:
{options}

Instruction:
1. Think step-by-step about the code/concept.
2. Eliminate wrong options.
3. Select the correct option.
4. Output STRICT JSON: {{"thought": "...", "final_answer": "Option"}}

Answer:"""

        # Generate
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1 # Low temp = More logic, less creative
        )
        result = self.tokenizer.batch_decode(outputs)[0]
        
        # Extract Answer
        try:
            # Look for the last JSON block
            match = re.search(r"\{.*\}", result.split("Answer:")[-1], re.DOTALL)
            if match:
                ans_data = json.loads(match.group(0))
                return ans_data.get("final_answer", "IDK")
            return "IDK"
        except:
            return "IDK"

# Simple test if running directly
if __name__ == "__main__":
    agent = AnswerAgent()
    sample = {
        "question": "What is the output of print(1 == True)?", 
        "options": ["A) True", "B) False", "C) Error", "D) 1"]
    }
    print(f"Test Answer: {agent.solve(sample)}")
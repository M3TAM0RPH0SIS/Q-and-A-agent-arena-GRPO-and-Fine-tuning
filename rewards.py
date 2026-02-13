import json
import re

# --- 1. Format Reward (The "Rules" Check) ---
def format_reward_func(completions, **kwargs):
    """Reward 1.0 if strict JSON, 0.0 otherwise."""
    rewards = []
    for content in completions:
        # Extract the content inside the first { and last }
        try:
            # Look for JSON structure
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue
            
            json_str = match.group(0)
            data = json.loads(json_str)
            
            # Check for required keys
            if all(k in data for k in ["question", "options", "answer", "explanation"]):
                rewards.append(1.0)
            else:
                rewards.append(0.5) # Partial credit
        except:
            rewards.append(0.0)
    return rewards

# --- 2. Complexity Reward (The "Difficulty" Check) ---
def complexity_reward_func(completions, **kwargs):
    """Reward longer, more complex questions."""
    rewards = []
    for content in completions:
        score = 0.0
        # Longer questions are usually harder
        if len(content) > 150: score += 0.2
        if len(content) > 300: score += 0.2
        
        # Technical keywords get points
        if "code" in content or "snippet" in content: score += 0.2
        if "Consider" in content or "Suppose" in content: score += 0.1
        
        rewards.append(score)
    return rewards

# --- 3. The "Judge" Reward (Placeholder for A-Agent) ---
def difficulty_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        rewards.append(0.0) 
    return rewards
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

class LocalHFWrapper:
    def __init__(self, model, tokenizer, conv_template=None,model_path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.conv_template = conv_template  # optional
        self.model_name = "qwen"
        # if model_path:
        #     name = Path(model_path).name.lower()
        #     if "qwen" in name:
        #         self.model_name = "qwen"
        #     elif "deepseek" in name:
        #         self.model_name = "deepseek"

    def __call__(self, prompts):
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
        return results if len(results) > 1 else results[0]
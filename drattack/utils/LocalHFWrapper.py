from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalHFWrapper:
    def __init__(self, model, tokenizer, conv_template=None):
        self.model = model
        self.tokenizer = tokenizer
        self.conv_template = conv_template  # optional
        self.model_name = "huggingface"

    def __call__(self, prompts):
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
        return results if len(results) > 1 else results[0]
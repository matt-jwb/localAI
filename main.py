import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

class LLM:
    def __init__(self, model_location, sys_prompt):
        self.model_path = get_latest_snapshot(model_location)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gptq = is_gptq_model(self.model_path)

        if self.is_gptq:
            if not torch.cuda.is_available():
                raise RuntimeError("[ERROR] No GPU found - cannot run GPTQ model")
            try:
                print("[SYSTEM] Attempting to load GPTQ quantized model with ExLLaMA...")
                self.model = AutoGPTQForCausalLM.from_quantized(self.model_path, use_safetensors=True, device_map="auto", use_triton=False, inject_fused_attention=False, use_exllama=True)
            except:
                print("[SYSTEM] ExLLaMA not supported. Loading GPTQ quantized model without ExLLaMA...")
                self.model = AutoGPTQForCausalLM.from_quantized(self.model_path, use_safetensors=True, device_map="auto", use_triton=False, inject_fused_attention=False)
        else:
            print("[SYSTEM] Loading standard model...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            self.model = torch.compile(self.model)
            self.model.to(self.device)

        self.max_possible_tokens = self.get_max_possible_tokens(self.model)
        self.max_new_tokens = min(512, self.max_possible_tokens // 4)
        self.max_length = self.max_possible_tokens - self.max_new_tokens

        self.conversation_history = []
        self.conversation_history.append(f"System: {sys_prompt}")

    def generate_response(self, prompt):
        self.conversation_history.append(f"User: {prompt}")
        self.trim_history_to_fit()
        full_prompt = "\n".join(self.conversation_history) + "\nAI:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True)
        # Setting tensors to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if self.is_gptq:
                outputs = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), max_new_tokens=self.max_new_tokens, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
            else:
                outputs = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), max_new_tokens=self.max_new_tokens, do_sample=True, no_repeat_ngram_size=2, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response.split("AI:")[-1].strip()

        self.conversation_history.append(f"AI: {response_text}")

        return response_text

def get_latest_snapshot(model_dir):
    subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not subdirs:
        return None
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return subdirs[0]

def get_path(loc):
    current_directory = os.getcwd()
    relative_path = os.path.join(current_directory, loc)
    normalised_path = os.path.normpath(relative_path)
    if os.path.commonpath([current_directory, normalised_path]) == current_directory:
        return normalised_path
    else:
        raise Exception("Invalid Location")

def is_gptq_model(model_path):
    return "gptq" in model_path.lower() or os.path.exists(os.path.join(model_path, "quantize_config.json"))

def main():
    location = input("Input model location >>  ")
    location = get_path(location)
    sys_prompt = "You are an AI Assistant, designed to help the user with a range of issues."
    model_instance = LLM(location, sys_prompt)

    text = input(">> ")
    while text != "quit":
        response = model_instance.generate_response(text)
        print(response)
        text = input(">> ")

main()
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

class LLM:
    def __init__(self, model_location, sys_prompt):
        self.model_path = self.get_latest_snapshot(model_location)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gptq = is_gptq_model(self.model_path)

        if self.is_gptq:
            if not torch.cuda.is_available():
                raise RuntimeError("[ERROR] No GPU found - cannot run GPTQ model")
            print("[SYSTEM] Loading GPTQ quantized model...")
            self.model = AutoGPTQForCausalLM.from_quantized(self.model_path, use_safetensors=True, device="cuda:0", use_triton=False, inject_fused_attention=False)
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

    def get_latest_snapshot(self, model_dir):
        for subdir in os.listdir(model_dir):
            subdir_path = os.path.join(model_dir, subdir)

            if os.path.isdir(subdir_path):
                snapshots_dir = os.path.join(subdir_path, 'snapshots')
                if os.path.isdir(snapshots_dir):
                    snapshot_folders = [f for f in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, f))]

                    latest_snapshot = max(snapshot_folders, key=lambda x: x)
                    return os.path.join(snapshots_dir, latest_snapshot)
        return None

    def get_max_possible_tokens(self, model):
        if hasattr(model.config, "max_position_embeddings"):
            return model.config.max_position_embeddings
        elif hasattr(model.config, "n_positions"):
            return model.config.n_positions
        elif hasattr(model.config, "max_sequence_length"):
            return model.config.max_sequence_length
        else:
            return 1024

    def trim_history_to_fit(self):
        full_prompt = "\n".join(self.conversation_history)
        tokens = self.tokenizer(full_prompt, return_tensors="pt")
        while tokens.input_ids.size(1) > self.max_length:
            # Pops the oldest message (apart from system prompt)
            self.conversation_history.pop(1)
            full_prompt = "\n".join(self.conversation_history)
            tokens = self.tokenizer(full_prompt, return_tensors="pt")

    def generate_response(self, prompt):
        self.conversation_history.append(f"User: {prompt}")
        self.trim_history_to_fit()
        full_prompt = "\n".join(self.conversation_history) + "\nAI:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True)
        # Setting tensors to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if self.is_gptq:
                outputs = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=self.max_new_tokens, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
            else:
                outputs = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=self.max_new_tokens, do_sample=True, no_repeat_ngram_size=2, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response.split("AI:")[-1].strip()

        self.conversation_history.append(f"AI: {response_text}")

        return response_text

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
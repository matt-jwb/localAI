import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self, model_location, sys_prompt):
        self.model_path = self.get_latest_snapshot(model_location)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
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

    def generate_response(self, prompt):
        self.conversation_history.append(f"User: {prompt}")
        full_prompt = "\n".join(self.conversation_history) + "\nAI:"

        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        attention_mask = torch.ones(inputs.shape, device=inputs.device)
        pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=512, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=pad_token_id)

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
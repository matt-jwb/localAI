import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_latest_snapshot(model_dir="./local_model/models--gpt2"):
    snapshots_dir = os.path.join(model_dir, "snapshots")
    snapshot_folders = [f for f in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, f))]

    latest_snapshot = max(snapshot_folders, key=lambda x: x)
    return os.path.join(snapshots_dir, latest_snapshot)

model_path = get_latest_snapshot()

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = torch.ones(inputs.shape, device=inputs.device)

    pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=pad_token_id)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


text = input(">> ")
while text != "quit":
    response = generate_response(text)
    print(response)

    text = input(">> ")
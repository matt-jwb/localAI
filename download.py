from transformers import AutoTokenizer, AutoModelForCausalLM

model = "gpt2"
AutoTokenizer.from_pretrained(model, cache_dir="./local_model")
AutoModelForCausalLM.from_pretrained(model, cache_dir="./local_model")

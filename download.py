import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_path(loc):
    current_directory = os.getcwd()
    relative_path = os.path.join(current_directory, loc)
    normalised_path = os.path.normpath(relative_path)
    if os.path.commonpath([current_directory, normalised_path]) == current_directory:
        return normalised_path
    else:
        raise Exception("Invalid Location")

def save_model(m, loc):
    try:
        AutoTokenizer.from_pretrained(m, cache_dir=loc)
        AutoModelForCausalLM.from_pretrained(m, cache_dir=loc)
    except Exception as ex:
        print(f"There was a problem downloading the model: {ex}")

def download():
    model = input("Input a model >>  ")
    location = input("Input save location >>  ")
    try:
        location = get_path(location)
    except Exception as e:
        print(f"There was a problem with the location: {e}")
    while model != "quit":
        save_model(model, location)
        model = input("Input a model >> ")
        location = input("Input save location >>  ")

download()

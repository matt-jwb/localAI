import os
from transformers import AutoTokenizer
from huggingface_hub import login, whoami, snapshot_download
from huggingface_hub.utils import LocalTokenNotFoundError, HfHubHTTPError


def get_path(loc):
    current_directory = os.getcwd()
    relative_path = os.path.join(current_directory, loc)
    normalised_path = os.path.normpath(relative_path)
    if os.path.commonpath([current_directory, normalised_path]) == current_directory:
        return normalised_path
    else:
        raise Exception("Invalid Location")

def save_model(model_name, cache_dir):
    try:
        print(f"Saving model: {model_name}")
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir, local_dir=os.path.join(cache_dir, model_name.replace("/", "_")))
        print(f"[SYSTEM] Model files downloaded to: {model_path}")
    except Exception as ex:
        print(f"There was a problem downloading the model: {ex}")

def is_logged_in():
    try:
        whoami()
        return True
    except (LocalTokenNotFoundError, HfHubHTTPError):
        return False

def try_login():
    token = input("Input your Hugging Face token >>  ")
    login(token=token, add_to_git_credential=True)

def download():
    if not is_logged_in():
        try_login()

    while True:
        model = input("Input a model >> ")
        if model.lower() == "quit":
            break

        location = input("Input save location >> ")
        try:
            location = get_path(location)
        except Exception as e:
            print(f"There was a problem with the location: {e}")
            continue

        save_model(model, location)

download()

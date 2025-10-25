# Local AI
This project allows the user to download and use a range of hugging face models on their local device

## Performance
Performance and results will vary drastically based on your hardware and choice of model. This project has currently been tested with TheBloke/Mistral-7B-Instruct-v0.2-GPTQ on an RTX 2080ti

## GPU usage
To ensure GPU is used you must have CUDA toolkit installed and a compatible GPU

## Usage
download.py will allow you to download a model
main.py will allow you to use the model
You must ensure your environment is set up correctly (I will be persuing ways to make this project ready to use, once I improve usability and performance)

## Dependencies
```
torch
transformers
auto_gptq
hf_xet (optional) - For improved model download speeds
optimum (optional) - Only needed for GPTQ models
```

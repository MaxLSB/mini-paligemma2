import os
import json
import glob
from safetensors import safe_open
from transformers import AutoTokenizer

from paligemma2.paligemma2 import PaliGemma2
from paligemma2.config_models import PaliGemma2Config
from paligemma.paligemma import PaliGemma
from paligemma.config_models import PaliGemmaConfig

################################### Load Hugging Face weights ###################################


def load_hf_model(model_path: str, model_type: str, device: str):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config

    if model_type == "paligemma":
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config_file = json.load(f)
            config = PaliGemmaConfig(**model_config_file)
        model = PaliGemma(config).to(device)

    elif model_type == "paligemma2":
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config_file = json.load(f)
            config = PaliGemma2Config(**model_config_file)
        model = PaliGemma2(config).to(device)

    else:
        raise ValueError(f"Model type {model_type} not supported.")

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)

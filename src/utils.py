import os
import json
import glob
from typing import Tuple
from safetensors import safe_open
from transformers import AutoTokenizer
from models.paligemma import PaliGemma, PaliGemmaConfig

################################### Load Hugging Face weights ###################################


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemma, AutoTokenizer]:
    # Load tokenizer (we don't code it here)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="right", local_files_only=True
    )
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # Load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemma(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)

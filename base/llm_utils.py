import torch 
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from transformers import T5EncoderModel, T5Config
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def last_token_pool(last_hidden_states, attention_mask, left_padding=False):
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.size(0)), sequence_lengths
    ]
    
def weighted_average_pool(last_hidden_states, attention_mask):
    seq_length = last_hidden_states.size(1)
    weights = (
        torch.arange(1, seq_length + 1, dtype=torch.float32)
        .unsqueeze(0)
        .to(last_hidden_states.device)
    )

    weighted_mask = weights * attention_mask.float()
    weighted_hidden_states = last_hidden_states * weighted_mask.unsqueeze(-1)

    sum_weighted_embeddings = torch.sum(weighted_hidden_states, dim=1)
    sum_weights = torch.sum(weighted_mask, dim=1, keepdim=True).clamp(min=1)

    weighted_average_embeddings = sum_weighted_embeddings / sum_weights

    return weighted_average_embeddings

@dataclass
class ModelConfig:
    name: str
    config_class: Optional[any] = None
    model_class: Optional[any] = None
    dropout_field: str = "dropout_rate"

MODEL_CONFIGS = {
    "t5-base": ModelConfig("t5-base", T5Config, T5EncoderModel),
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": ModelConfig(
        "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        T5Config,
        T5EncoderModel,
    ),
}

def get_model_and_tokenizer(model_name: str, device: str='cpu'):
    # cache_dir="./from_pretrained", local_files_only=True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False, cache_dir="./from_pretrained", local_files_only=False
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        config = model_config.config_class.from_pretrained(model_name, cache_dir="./from_pretrained", local_files_only=False)
        setattr(config, model_config.dropout_field, 0)
        model = model_config.model_class.from_pretrained(
            model_name, config=config, cache_dir="./from_pretrained", local_files_only=False
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            model_name, device_map=device, trust_remote_code=True, cache_dir="./from_pretrained", local_files_only=False
        )

    return model, tokenizer

def cache__get_model_and_tokenizer(model_name: str, device: str='cpu'):
    # cache_dir="./from_pretrained", local_files_only=True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False, cache_dir="./from_pretrained"
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        config = model_config.config_class.from_pretrained(model_name, cache_dir="./from_pretrained")
        setattr(config, model_config.dropout_field, 0)
        model = model_config.model_class.from_pretrained(
            model_name, config=config, cache_dir="./from_pretrained"
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            model_name, device_map=device, trust_remote_code=True, cache_dir="./from_pretrained"
        )

    return model, tokenizer


def get_huggingface_embeddings(
    texts,
    model_name="tiiuae/falcon-7b",
    max_length=512,
    batch_size=8,
    pooling_method="cls",
    prefix=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    normalize_embeddings=False,
):
    """
    General function to get embeddings from a HuggingFace transformer model.
    """
    print(f"featurizing with {model_name}")
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    left_padding = tokenizer.padding_side == "left"
    model.eval()

    # optionally add prefix to each text
    if prefix:
        texts = [prefix + text for text in texts]

    pooling_functions = {
        "average": average_pool,
        "cls": lambda x, _: x[:, 0],
        "last_token_pool": partial(last_token_pool, left_padding=left_padding),
        "weighted_average": weighted_average_pool,
    }

    embeddings_list = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
    ):
        batch_texts = texts[i : i + batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded_input)
            pooled = pooling_functions[pooling_method](
                outputs.last_hidden_state, encoded_input["attention_mask"]
            )

            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            embeddings_list.append(pooled.cpu().numpy())

        torch.cuda.empty_cache()

    return np.concatenate(embeddings_list, axis=0)

import os
import torch
from copy import deepcopy
from open_clip import create_model_from_pretrained, get_tokenizer


class BiomedCLIPTokenizerWrapper:
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        max_length: int = 77,
    ):
        self.tokenizer = get_tokenizer(model_name)
        self.model_max_length = max_length

    def __call__(self, text, max_length=None, *args, **kwargs):
        input_ids = self.tokenizer(text, max_length)
        return {"input_ids": input_ids}


class BiomedCLIPTextEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ):
        super().__init__()
        model, _ = create_model_from_pretrained(
            model_name,
            device="cpu",
            cache_dir=os.path.expanduser("~/.cache"),
        )
        self.pad_token = model.text.config.pad_token_id
        self.text_encoder = deepcopy(model.text.transformer)
        del model

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token).long()
        return self.text_encoder(input_ids, attention_mask)

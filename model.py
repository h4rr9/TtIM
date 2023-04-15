"""Model for text to image to text transformer.

The transformer models images as language tokens and is trained on LM loss.
The image is quantized into 8 levels and assigned a token for RGB values
for a total of 8*8*8=512 new special tokens
"""
import torch
import torch.nn as nn

from transformers import (
    OPTForCausalLM,
    AutoTokenizer,
)
from typing import Optional, Tuple
from enum import Enum


IMAGE_SIZE = 32

PE_TYPES = Enum("PE_TYPES", ["VIT", "RWKV"])


class Tim(OPTForCausalLM):
    """Model for text to image to text transformer."""

    _keys_to_ignore_on_load_missing = [
        r"pe",
        r"pe_x",
        r"pe_y",
    ] + OPTForCausalLM._keys_to_ignore_on_load_missing

    def __init__(self, config, pe_type: PE_TYPES = PE_TYPES.VIT):
        """Initialize the model."""
        assert pe_type in PE_TYPES, "Unrecognised position embedding type"
        super(Tim, self).__init__(config)

        embed_size = config.hidden_size
        # OPT uses init_std instead of initializer_range
        init_std = config.init_std
        self.pe_type = pe_type
        # positional embeddings of size (num_pixels x embed)
        self.pe_x: nn.Parameter = nn.Parameter(torch.randn(1, IMAGE_SIZE, embed_size))
        self.pe_y: nn.Parameter = nn.Parameter(torch.randn(IMAGE_SIZE, 1, embed_size))
        self.pe_x.data.normal_(mean=0.0, std=init_std)
        self.pe_y.data.normal_(mean=0.0, std=init_std)

        self.init_weights()

    def prepare_inputs(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.TensorType, torch.TensorType]:
        """Embed image positional embeddings."""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids \
                and inputs_embds at the same time"
            )
        elif input_ids is not None:
            # retrieve token embeddings
            emb = self.get_input_embeddings()
            inputs_embeds = emb(input_ids)

        if image_masks is not None:
            batch, sequence_len, embed_dim = inputs_embeds.size()
            # retrieve embeddings of images in the sequence
            # reshape for broadcasting
            image_embeddings = inputs_embeds[image_masks].view(
                batch, IMAGE_SIZE * IMAGE_SIZE, embed_dim
            )
            # add image positional embeddings

            image_embeddings += (self.pe_x + self.pe_y).view(
                IMAGE_SIZE * IMAGE_SIZE, self.config.hidden_size
            )
        return inputs_embeds


if __name__ == "__main__":
    model = Tim.from_pretrained("facebook/opt-125m")

    x = model.pe_x.sum()

    model.save_pretrained("test_opt_model")

    model = Tim.from_pretrained("test_opt_model")

    y = model.pe_x.sum()

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    inputs = tokenizer("To be or not to be that is the", return_tensors="pt")

    outputs = model.generate(**inputs, max_length=20)

    print(tokenizer.batch_decode(outputs))

    print(x, y)

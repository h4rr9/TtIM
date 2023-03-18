"""Model for text to image to text transformer.

The transformer models images as language tokens and is trained on LM loss.
The image is quantized into 8 levels and assigned a token for RGB values
for a total of 8*8*8=512 new special tokens
"""
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GPTNeoForCausalLM
from typing import Optional
from enum import Enum

GPT_NEO_LARGE: str = "EleutherAI/gpt-neo-2.7B"
GPT_NEO_MEDIUM: str = "EleutherAI/gpt-neo-1.3B"
GPT_NEO_SMALL: str = "EleutherAI/gpt-neo-125M"

IMAGE_SIZE = 32

PE_TYPES = Enum("PE_TYPES", ["VIT", "RWKV"])


class Tim(GPTNeoForCausalLM):
    """Model for text to image to text transformer."""

    _keys_to_ignore_on_load_missing = [
        r"pe"
    ] + GPTNeoForCausalLM._keys_to_ignore_on_load_missing

    def __init__(self, config, pe_type: PE_TYPES = PE_TYPES.VIT):
        """Initialize the model."""
        assert pe_type in PE_TYPES, "Unrecognised position embedding type"
        super(Tim, self).__init__(config)

        embed_size = config.hidden_size

        if pe_type == PE_TYPES.RWKV:
            # positional embeddings of size (num_pixels x embed)
            self.pe_x: torch.Tensor = nn.Parameter(
                torch.randn(1, IMAGE_SIZE, embed_size)
            )
            self.pe_y: torch.Tensor = nn.Parameter(
                torch.randn(IMAGE_SIZE, 1, embed_size)
            )
            self.pe_x.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.pe_y.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.pe: torch.Tensor = (self.pe_x + self.pe_y).view(
                IMAGE_SIZE * IMAGE_SIZE, -1
            )
        elif pe_type == PE_TYPES.VIT:
            self.pe: torch.Tensor = nn.Parameter(
                torch.randn(IMAGE_SIZE * IMAGE_SIZE, embed_size)
            )
            self.pe.data.normal_(mean=0.0, std=self.config.initializer_range)

        self.init_weights()

        # self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

        # pixel_tokens = [
        #     f"[{r}{g}{b}]" for r in range(8) for g in range(8) for b in range(8)
        # ]
        # self.tokenizer.add_tokens(pixel_tokens)
        # self.tokenizer.add_tokens(["[Text]", "[Image]", "[TextFirst]", "[ImageFirst]"])
        # self.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_image_masks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None
    ) -> CausalLMOutputWithPast:
        """Forward pass of model."""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids \
                and inputs_embds at the same time"
            )

        elif input_ids is not None:
            # retrieve token embeddings
            inputs_embeds = self.wte(input_ids)

        batch, sequence_len, embed_dim = inputs_embeds.size()
        # retrieve embeddings of images in the sequence
        # reshape for broadcasting
        image_embeddings = inputs_embeds[input_image_masks].view(
            batch, IMAGE_SIZE * IMAGE_SIZE, embed_dim
        )
        # add image positional embeddings
        image_embeddings += self.pe

        # compute transformer outputs
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache
        )

        return outputs


if __name__ == "__main__":
    tim = Tim.from_pretrained(GPT_NEO_SMALL)

    em_mean_first = tim.get_input_embeddings().weight.data
    lm_head_first = tim.get_output_embeddings().weight.data

    another_tim_small = GPTNeoForCausalLM.from_pretrained(GPT_NEO_SMALL)

    em_mean_second = another_tim_small.get_input_embeddings().weight.data
    lm_head_second = another_tim_small.get_output_embeddings().weight.data

    assert torch.all(
        torch.isclose(em_mean_first, em_mean_second)
    ), "Something went wrong, model weights are not equal."

    assert torch.all(
        torch.isclose(lm_head_first, lm_head_second)
    ), "Something went wrong, model weights are not equal."

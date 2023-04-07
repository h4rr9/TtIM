"""Model for text to image to text transformer.

The transformer models images as language tokens and is trained on LM loss.
The image is quantized into 8 levels and assigned a token for RGB values
for a total of 8*8*8=512 new special tokens
"""
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    GPTNeoForCausalLM,
    OPTForCausalLM,
    BloomForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
)
from typing import Optional, Tuple
from enum import Enum


class TimModels:
    """Namespace the decoders of Tim models."""

    GPT_NEO_LARGE: str = "EleutherAI/gpt-neo-2.7B"
    GPT_NEO_MEDIUM: str = "EleutherAI/gpt-neo-1.3B"
    GPT_NEO_SMALL: str = "EleutherAI/gpt-neo-125M"

    OPT_LARGE: str = "facebook/opt-2.7B"
    OPT_MEDIUM: str = "facebook/opt-1.3B"
    OPT_SMALL: str = "facebook/opt-125M"

    BLOOM_SMALL: str = "bigscience/bloom-560m"
    BLOOM_MEDIUM: str = "bigscience/bloom-1.1b"
    BLOOM_LARGE: str = "bigscience/bloom-3b"


IMAGE_SIZE = 32

PE_TYPES = Enum("PE_TYPES", ["VIT", "RWKV"])


def tim_factory(CausalModel: PreTrainedModel):
    """Initialize class with specified parent class."""

    class Tim(CausalModel):
        """Model for text to image to text transformer."""

        _keys_to_ignore_on_load_missing = [
            r"pe",
            r"pe_x",
            r"pe_y",
        ] + CausalModel._keys_to_ignore_on_load_missing

        def __init__(self, config, pe_type: PE_TYPES = PE_TYPES.VIT):
            """Initialize the model."""
            assert pe_type in PE_TYPES, "Unrecognised position embedding type"
            super(Tim, self).__init__(config)

            self._name = CausalModel.__name__

            embed_size = config.hidden_size
            # OPT uses init_std instead of initializer_range
            init_std = (
                config.init_std
                if CausalModel is OPTForCausalLM
                else self.config.initializer_range
            )
            self.pe_type = pe_type
            if self.pe_type == PE_TYPES.RWKV:
                # positional embeddings of size (num_pixels x embed)
                self.pe_x: nn.Parameter = nn.Parameter(
                    torch.randn(1, IMAGE_SIZE, embed_size)
                )
                self.pe_y: nn.Parameter = nn.Parameter(
                    torch.randn(IMAGE_SIZE, 1, embed_size)
                )
                self.pe_x.data.normal_(mean=0.0, std=init_std)
                self.pe_y.data.normal_(mean=0.0, std=init_std)
            elif pe_type == PE_TYPES.VIT:
                self.pe: nn.Parameter = nn.Parameter(
                    torch.randn(IMAGE_SIZE * IMAGE_SIZE, embed_size)
                )
                self.pe.data.normal_(mean=0.0, std=init_std)

            self.init_weights()

        def prepare_inputs(
            self,
            input_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            input_image_masks: Optional[torch.Tensor] = None,
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

            if input_image_masks is not None:
                batch, sequence_len, embed_dim = inputs_embeds.size()
                # retrieve embeddings of images in the sequence
                # reshape for broadcasting
                image_embeddings = inputs_embeds[input_image_masks].view(
                    batch, IMAGE_SIZE * IMAGE_SIZE, embed_dim
                )
                # add image positional embeddings

                if self.pe_type == PE_TYPES.VIT:
                    image_embeddings += self.pe
                else:
                    image_embeddings += (self.pe_x + self.pe_y).view(
                        IMAGE_SIZE * IMAGE_SIZE, self.config.hidden_size
                    )
            return (input_ids, inputs_embeds)


        def prepare_input(
                self,
                input_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                input_image_masks: Optional[torch.Tensor] = None,
        ):




    return Tim


def get_tim(model_name, positional_embedding: PE_TYPES, device="cpu"):
    """Return specified pretrained model."""
    model = None
    if "opt" in model_name:
        model = tim_factory(OPTForCausalLM)
    elif "bloom" in model_name:
        model = tim_factory(BloomForCausalLM)
    elif "gpt-neo" in model_name:
        model = tim_factory(GPTNeoForCausalLM)
    else:
        raise ValueError("Unknown model_name.")

    return model.from_pretrained(model_name, pe_type=positional_embedding).to(device)


if __name__ == "__main__":
    model = get_tim(TimModels.BLOOM_SMALL, PE_TYPES.RWKV, "cpu")

    x = model.pe_x.sum()

    model.save_pretrained("test_opt_model")

    model = tim_factory(BloomForCausalLM).from_pretrained(
        "test_opt_model", PE_TYPES.RWKV
    )

    y = model.pe_x.sum()

    tokenizer = AutoTokenizer.from_pretrained(TimModels.BLOOM_SMALL)

    inputs = tokenizer("To be or not to be that is the", return_tensors="pt")

    outputs = model.generate(**inputs, max_length=20)

    print(tokenizer.batch_decode(outputs))

    print(model._name, x, y)

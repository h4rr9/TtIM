#!/usr/bin/env python3
import numpy as np
from typing import Callable, List, NewType, Any, Dict
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import torch
from enum import Enum


# from torch.utils.data import Dataset as TorchDataset
# from datasets import Dataset as Dataset, load_dataset
from torch.utils.data import DataLoader
from datasets import load_dataset

IMAGE_DIM = 32
IMAGE_LEN = IMAGE_DIM * IMAGE_DIM


InputDataClass = NewType("InputDataClass", Any)


IMAGE_TOKEN = "[Image]"
IMAGE_FIRST_TOKEN = "[ImageFirst]"
TEXT_TOKEN = "[Text]"
TEXT_FIRST_TOKEN = "[TextFirst]"


PromptType = Enum("PrompType", ["ImagePrompt", "TextPrompt"])


def get_custom_collater(
    tokenizer: Tokenizer, rng: np.random.Generator, p: float = 0.5
) -> Callable:
    def _collater(
        features: List[InputDataClass],
    ) -> Dict[str, Any]:
        """Generate tokenized training data.

        First generate data with special tokens randomly.
        Second create image mask from the generated data.
        Third Tokenize the generated data.


        input:
        batch_images: batch of strings
        batch_captions: batch of captions (either strings or list of strings)
        p: percentage of Prompts of type TextPrompt, should be in [0.0, 1.0)
        """
        batch = {}

        assert 0.0 <= p <= 1.0, f"p should be in [0.0, 1.0], got {p}"

        prompts, image_masks = [], []
        image_start_positions = []

        batch_size = len(features)

        coins = np.zeros(shape=batch_size, dtype=bool)
        coins[: int(p * len(batch_size))] = True
        np.random.shuffle(coins)

        batch_images = [b["palette_images"] for b in features]
        batch_captions = [b["captions"] for b in features]

        kinds = []
        for coin_toss, image, captions in zip(coins, batch_images, batch_captions):
            # coin_toss = rng.random() > (1.0 - p)

            caption = captions if type(captions) is str else rng.choice(captions)

            prompt = (
                f"{TEXT_FIRST_TOKEN}{caption}{IMAGE_TOKEN}{image}"
                if coin_toss
                else f"{IMAGE_FIRST_TOKEN}{image}{TEXT_TOKEN}{caption}"
            )

            image_start_position = (
                3
                + len(
                    tokenizer.tokenize(caption)  # does not count </s>
                )  # 3 corresponds to </s> and [TextFirst] and [Image]
                if coin_toss
                else 2  # 2 corresconds to </s> and [ImageFirst]
            )

            kinds.append(1 if coin_toss else 0)
            image_start_positions.append(image_start_position)
            prompts.append(prompt)

        prompts = tokenizer(prompts, return_tensors="pt", padding=True)
        batch["input_ids"] = prompts["input_ids"]
        batch["attention_mask"] = prompts["attention_mask"]

        image_masks = torch.zeros(size=batch["attention_mask"].shape, dtype=bool)

        for image_mask, image_start_position in zip(image_masks, image_start_positions):
            image_mask[image_start_position : image_start_position + IMAGE_LEN] = True

        batch["image_masks"] = image_masks
        batch["kinds"] = torch.tensor(kinds, dtype=torch.long)

        return batch

    return _collater


def prepare_tokenizer(tokenizer: Tokenizer, args) -> Tokenizer:
    """Prepare tokenizer."""
    if args.dataset_name.endswith("_1_bit"):
        num_colors = 2
        zero_fill = 0
    elif args.dataset_name.endswith("_9_bit"):
        num_colors = 512
        zero_fill = 3
    elif args.dataset_name.endswith("_12_bit"):
        num_colors = 4098
        zero_fill = 4
    else:
        raise ValueError(
            f"Dataset expected to end with _1_bit, _9_bit, or _12_bit, got {args.dataset_name}"
        )

    pixel_tokens = ["[" + str(i).zfill(zero_fill) + "]" for i in range(num_colors)]
    tokenizer.add_tokens(pixel_tokens)
    tokenizer.add_tokens([IMAGE_FIRST_TOKEN, TEXT_FIRST_TOKEN, IMAGE_TOKEN, TEXT_TOKEN])
    return tokenizer


if __name__ == "__main__":
    d = load_dataset("h4rr9/cifar_10_palette")

    rng = np.random.default_rng()
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained("facebook/opt-125m"))

    c = get_custom_collater(tok, rng, p=0.75)

    dl = DataLoader(d["train"], shuffle=True, collate_fn=c, batch_size=2)

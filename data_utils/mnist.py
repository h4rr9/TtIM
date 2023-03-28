#!/usr/bin/env python3
"""Utilities to process mnist dataset."""
from datasets import load_dataset, Features, Value, Array2D, DatasetDict, Sequence
from PIL import Image
import numpy as np


number_to_text = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def add_captions_and_resize(example):
    image = example["image"].convert("RGB").resize((32, 32), Image.LANCZOS)
    image = np.array(image, dtype=np.uint8).reshape(-1, 3)
    label = example["label"]

    return {"images": image, "captions": [str(label), number_to_text[label]]}


if __name__ == "__main__":
    mnist = load_dataset("mnist")

    features = Features(
        {
            "images": Array2D(shape=(1024, 3), dtype="uint8"),
            "captions": Sequence(
                feature=Value("string"),
                length=2,
            ),
        }
    )

    mnist_captions_train = mnist["train"].map(
        add_captions_and_resize,
        remove_columns=mnist["train"].column_names,
        features=features,
    )
    mnist_captions_val = mnist["test"].map(
        add_captions_and_resize,
        remove_columns=mnist["test"].column_names,
        features=features,
    )

    full_mnist = DatasetDict()
    full_mnist["train"] = mnist_captions_train
    full_mnist["validation"] = mnist_captions_val

    full_mnist.push_to_hub("h4rr9/mnist_32x32", private=True)

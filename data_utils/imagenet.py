#!/usr/bin/env python3
"""Utilities for creating imagenet captions dataset."""

import numpy as np
import json
from tqdm.auto import tqdm
from multiprocessing import Pool

from datasets import Dataset, DatasetDict, Features, Value, Array2D


def _create_imagenet_dataset(imagenet_npz_file: str, simple_labels: str, split: str):
    """Create imagenet dataset."""
    batch_size = 1000

    with np.load(imagenet_npz_file) as imagenet:
        labels = imagenet["labels"]
        data = imagenet["data"].reshape(-1, 1024, 3).astype(np.uint8)

        n_batches = (
            data.shape[0] // batch_size + 0 if data.shape[0] % batch_size == 0 else 1
        )

        batched_label_data = (
            labels[i : i + batch_size] for i in range(data.shape[0], step=batch_size)
        )

        batched_image_data = (
            data[i : i + batch_size] for i in range(data.shape[0], step=batch_size)
        )

        with open(simple_labels, "r") as simple_labels_file:
            simple_labels = json.load(simple_labels_file)

        def _apply(labels):
            return [simple_labels[int(label)] for label in labels]

        with Pool(processes=12) as pool:
            with tqdm(
                total=n_batches, desc=f"Creating Imagenet {split} dataset"
            ) as p_bar:
                for images, label_result in zip(
                    batched_image_data, pool.imap(_apply, batched_label_data)
                ):
                    for image, simple_label in zip(images, label_result):
                        p_bar.update()
                        yield {"images": image, "captions": simple_label}


if __name__ == "__main__":
    features = Features(
        {"images": Array2D(shape=(1024, 3), dtype="uint8"), "captions": Value("string")}
    )

    train_dataset = Dataset.from_generator(
        generator=_create_imagenet_dataset,
        features=features,
        gen_kwargs={
            "imagenet_npz_file": "./data/imagenet/Imagenet32_train_npz/train_data.npz",
            "simple_labels": "./data/imagenet/imagenet-simple-labels.json",
            "split": "train",
        },
    )
    val_dataset = Dataset.from_generator(
        generator=_create_imagenet_dataset,
        features=features,
        gen_kwargs={
            "imagenet_npz_file": "./data/imagenet/Imagenet32_val_npz/val_data.npz",
            "simple_labels": "./data/imagenet/imagenet-simple-labels.json",
            "split": "validation",
        },
    )

    train_dataset.set_format(
        "numpy", columns=["images"], format_kwargs={"dtype": np.uint8}
    )
    val_dataset.set_format(
        "numpy", columns=["images"], format_kwargs={"dtype": np.uint8}
    )

    full_dataset = DatasetDict()
    full_dataset["train"] = train_dataset
    full_dataset["validation"] = val_dataset

    full_dataset.push_to_hub("h4rr9/imagenet_32x32", private=True)

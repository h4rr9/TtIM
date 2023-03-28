#!/usr/bin/env python3
"""Utilities for creating imagenet captions dataset."""

import numpy as np
import json
from tqdm.auto import tqdm
from multiprocessing import Pool
import os
import re

from datasets import Dataset, DatasetDict, Features, Value, Array2D


shared_lookup_table = []


def _init_worker(shared_lookup_table):
    global lookup_table
    lookup_table = shared_lookup_table


def _apply(tuple_args):
    global lookup_table
    return [(data, lookup_table[int(label) - 1]) for (data, label) in zip(*tuple_args)]


# taken from https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    """Return an int if possible, or unchanged."""
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_keys(s):
    """Turn a string into a list of string and number chunks."""
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def _create_imagenet_dataset(
    imagenet_npz_folder: str, simple_labels_file: str, split: str
):
    """Create imagenet dataset."""
    if not imagenet_npz_folder.endswith("/"):
        imagenet_npz_folder += "/"

    with open(simple_labels_file, "r") as simple_labels_reader:
        global shared_lookup_table
        shared_lookup_table = json.load(simple_labels_reader)

    batch_size = 100
    for npz_file in sorted(os.listdir(imagenet_npz_folder), key=alphanum_keys):
        if npz_file.endswith(".npz"):
            batch_id = npz_file.split("_")[-1][:-4]
            imagenet_npz_batch_file = imagenet_npz_folder + npz_file
            with np.load(imagenet_npz_batch_file) as imagenet_batch:
                labels = imagenet_batch["labels"]
                data = imagenet_batch["data"].astype(np.uint8)

                data = np.dstack(
                    [data[:, :1024], data[:, 1024:2048], data[:, 2048:3072]]
                ).reshape(-1, 1024, 3)

                assert (
                    labels.shape[0] == data.shape[0]
                ), "expected same number of data and label samples, got {data.shape[0]} and {labels.shape[0]}"

                n_batches = data.shape[0] // batch_size + (
                    0 if data.shape[0] % batch_size == 0 else 1
                )

                batched_data = (
                    (data[i : i + batch_size], labels[i : i + batch_size])
                    for i in range(0, data.shape[0], batch_size)
                )

                with Pool(
                    processes=24,
                    initializer=_init_worker,
                    initargs=(shared_lookup_table,),
                ) as pool:
                    with tqdm(
                        total=n_batches,
                        desc=f"Creating Imagenet {split} dataset from batch {batch_id}",
                    ) as p_bar:
                        for result in pool.imap_unordered(_apply, batched_data):
                            p_bar.update()
                            for image, simple_label in result:
                                yield {"images": image, "captions": simple_label}


if __name__ == "__main__":
    features = Features(
        {"images": Array2D(shape=(1024, 3), dtype="uint8"), "captions": Value("string")}
    )

    train_dataset = Dataset.from_generator(
        generator=_create_imagenet_dataset,
        features=features,
        gen_kwargs={
            "imagenet_npz_folder": "./data/imagenet/Imagenet32_train_npz/",
            "simple_labels_file": "./data/imagenet/imagenet-simple-labels-fixed.json",
            "split": "train",
        },
    )
    val_dataset = Dataset.from_generator(
        generator=_create_imagenet_dataset,
        features=features,
        gen_kwargs={
            "imagenet_npz_folder": "./data/imagenet/Imagenet32_val_npz/",
            "simple_labels_file": "./data/imagenet/imagenet-simple-labels-fixed.json",
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

#!/usr/bin/env python3

import numpy as np
import pickle
import re

from tqdm.auto import tqdm

from datasets import Dataset, DatasetDict, Features, Value, Array2D
from glob import glob


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


def _create_cifar_dataset(data_file_prefix: str, labels_file: str, split: str):
    """Create cifar dataset."""
    assert split in (
        "train",
        "validation",
    ), f"split must be either 'train' or 'validation' got {split}"

    with open(labels_file, "rb") as f:
        labels_dict = pickle.load(f, encoding="bytes")

    # convert byte strings to strings
    labels_dict = [label.decode() for label in labels_dict[b"label_names"]]

    for file in sorted(glob(data_file_prefix + "*"), key=alphanum_keys):
        with open(file, "rb") as f:
            data = pickle.load(f, encoding="bytes")

            # 10000 x 3072
            images_data = data[b"data"]
            labels_data = data[b"labels"]

            images_data = np.dstack(
                [
                    images_data[:, :1024],
                    images_data[:, 1024:2048],
                    images_data[:, 2048:3072],
                ]
            ).reshape(-1, 1024, 3)

            assert (
                len(labels_data) == images_data.shape[0]
            ), "expected same number of data and label samples, got {data.shape[0]} and {labels.shape[0]}"

            for image_data, label_data in tqdm(zip(images_data, labels_data)):
                yield {"images": image_data, "captions": labels_dict[label_data]}


if __name__ == "__main__":
    features = Features(
        {"images": Array2D(shape=(1024, 3), dtype="uint8"), "captions": Value("string")}
    )

    train_dataset = Dataset.from_generator(
        generator=_create_cifar_dataset,
        features=features,
        gen_kwargs={
            "data_file_prefix": "./data/cifar/cifar-10-batches-py/data_batch",
            "labels_file": "./data/cifar/cifar-10-batches-py/batches.meta",
            "split": "train",
        },
    )

    validation_dataset = Dataset.from_generator(
        generator=_create_cifar_dataset,
        features=features,
        gen_kwargs={
            "data_file_prefix": "./data/cifar/cifar-10-batches-py/test_batch",
            "labels_file": "./data/cifar/cifar-10-batches-py/batches.meta",
            "split": "train",
        },
    )

    train_dataset.set_format(
        "numpy", columns=["images"], format_kwargs={"dtype": np.uint8}
    )
    validation_dataset.set_format(
        "numpy", columns=["images"], format_kwargs={"dtype": np.uint8}
    )

    full_dataset = DatasetDict()
    full_dataset["train"] = train_dataset
    full_dataset["validation"] = validation_dataset

    full_dataset.push_to_hub("h4rr9/cifar_10_32x32", private=True)

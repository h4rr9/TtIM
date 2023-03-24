#!/usr/bin/env python3
"""Utilities for creating mscoco dataset."""

import json
from collections import defaultdict
from tqdm.auto import tqdm
from more_itertools import batched


from multiprocess import Pool
from PIL import Image
import numpy as np

from datasets import Features, Sequence, Array2D, Value, Dataset, DatasetDict


def _create_image_annotation_pairs(annotation_file: str, images_path: str, split: str):
    """Create image annotation pairs."""
    if not images_path.endswith("/"):
        images_path += "/"
    pairs = defaultdict(list)
    images = dict()
    with open(annotation_file, "r") as ann_file:
        annotations = json.load(ann_file)

        for caption in tqdm(annotations["annotations"], desc=f"Creating {split} pairs"):
            if (
                len(pairs[caption["image_id"]]) < 5
            ):  # limit to 5 captions per image (6,7 exist)
                pairs[caption["image_id"]].append(caption["caption"])
        for image in tqdm(annotations["images"], desc=f"Cacheing {split} images"):
            [image_name, _] = image["file_name"].split(".")
            images[image["id"]] = image_name

    batched_data = batched(pairs.items(), 1000)

    def _apply(pairs):
        results = []
        for image_id, captions in pairs:
            with open(
                images_path + str(images[image_id]).zfill(12), "rb"
            ) as image_file:
                image_bytes = image_file.read()
                image = np.array(
                    Image.frombytes(mode="RGB", data=image_bytes, size=(32, 32)),
                    dtype=np.uint8,
                ).reshape(-1, 3)

                results.append((image, captions))
        return results

    with Pool(processes=12) as pool:
        with tqdm(total=len(pairs), desc=f"Creating {split} dataset") as p_bar:
            for result in pool.imap_unordered(_apply, batched_data):
                for image, captions in result:
                    p_bar.update()
                    yield {"images": image, "captions": captions}


if __name__ == "__main__":
    features = Features(
        {
            "images": Array2D(shape=(1024, 3), dtype="uint8"),
            "captions": Sequence(feature=Value("string"), length=5),
        }
    )

    train_dataset = Dataset.from_generator(
        generator=_create_image_annotation_pairs,
        gen_kwargs={
            "annotation_file": "./data/mscoco/annotations/captions_train2017.json",
            "images_path": "./data/mscoco/train_32x32/",
            "split": "train",
        },
    )
    val_dataset = Dataset.from_generator(
        generator=_create_image_annotation_pairs,
        gen_kwargs={
            "annotation_file": "./data/mscoco/annotations/captions_val2017.json",
            "images_path": "./data/mscoco/val_32x32/",
            "split": "validation",
        },
    )

    val_dataset.set_format(
        "numpy", columns=["images"], format_kwargs={"dtype": np.uint8}
    )

    train_dataset.set_format(
        "numpy", columns=["images"], format_kwargs={"dtype": np.uint8}
    )

    full_dataset = DatasetDict()
    full_dataset["train"] = train_dataset
    full_dataset["validaton"] = val_dataset

    full_dataset.push_to_hub("h4rr9/mscoco_32x32", private=True)

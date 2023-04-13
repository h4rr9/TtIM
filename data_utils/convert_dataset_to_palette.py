#!/usr/bin/env python3

from datasets import load_dataset
import numpy as np
from scipy.spatial import cKDTree
from functools import partial
import argparse
# from utils import PIXEL_TOKENS


def _apply(examples, kdt, sort_idx, image_tokens_dict):
    images = examples["images"]

    batch_size, num_pixels, _ = images.shape
    imgs = images.reshape(-1, 3)
    # palette idxs for every pixel
    _, imgs_palette_idx = kdt.query(imgs, k=1)
    # reshape into batches of images pixel idxs
    imgs_palette_idx = imgs_palette_idx.reshape(batch_size, num_pixels)

    token_idx = np.searchsorted(list(image_tokens_dict.keys()),
                                imgs_palette_idx,
                                sorter=sort_idx)
    palette_images = np.asarray(list(
        image_tokens_dict.values()))[sort_idx][token_idx]

    palette_images = ["".join(palette_img) for palette_img in palette_images]

    return {"palette_images": palette_images, "captions": examples["captions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Huggingface dataset to convert and upload, example: mnist_32x32",
    )

    parser.add_argument(
        "--user",
        "-u",
        type=str,
        default="h4rr9",
        help="Huggingface username",
    )

    parser.add_argument(
        "--palette",
        "-o",
        type=str,
        default="./assets/9-bit.npy",
        help="Huggingface username",
    )

    parser.add_argument(
        "--upload_name",
        "-n",
        type=str,
        required=True,
        help="Name of new dataset in Huggingface",
    )
    args = parser.parse_args()

    assert (
        args.dataset != args.upload_name
    ), "This will rewrite existing dataset aborting, use different new dataset name"

    palette = np.load(args.palette)

    if len(palette) == 512:
        PIXEL_TOKENS = [f"[{i:0>3}]" for i in range(len(palette))]
    elif len(palette) == 4096:
        PIXEL_TOKENS = [f"[{i:0>4}]" for i in range(len(palette))]
    elif len(palette) == 2:
        PIXEL_TOKENS = [f"[{i:0>1}]" for i in range(len(palette))]
    else:
        raise ValueError("Unexpected palette length")

    image_tokens_dict = {
        idx: img_token
        for idx, img_token in enumerate(PIXEL_TOKENS)
    }

    kdt = cKDTree(palette)

    dataset = load_dataset(f"{args.user}/{args.dataset}")
    dataset.set_format("numpy",
                       columns=["images"],
                       format_kwargs={"dtype": np.uint8})

    sort_idx = np.argsort(list(image_tokens_dict.keys()))

    apply = partial(_apply,
                    kdt=kdt,
                    sort_idx=sort_idx,
                    image_tokens_dict=image_tokens_dict)

    new_dataset = dataset.map(apply,
                              batched=True,
                              remove_columns=["images"],
                              batch_size=10000)

    new_dataset.push_to_hub(f"{args.user}/{args.upload_name}", private=True)

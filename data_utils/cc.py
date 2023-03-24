"""Utility functions."""
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import csv
from datasets import Dataset, Features, Value, Array2D, DatasetDict

CC_TRAIN_SIZE = 3318333
CC_VAL_SIZE = 15840

CC_PATH = "./data/cc/"


def create_cc_dataset_split(report_file: str, captions_file: str, split: str):
    """Run main tsv process."""
    with open(report_file, "r") as reports, open(captions_file, "r") as captions:
        captions_reader = csv.reader(captions, delimiter="\t")
        report_reader = csv.reader(reports, delimiter="\t")

        for caption_row, report_row in tqdm(
            zip(captions_reader, report_reader),
            total=CC_TRAIN_SIZE if split == "train" else CC_VAL_SIZE,
        ):
            if "200" in report_row:
                assert caption_row[1] == report_row[5]

                with open(CC_PATH + report_row[0], "rb") as f:
                    image_bytes = f.read()

                    if int(float(report_row[3])) == 3072:  # 3 bytes per pixel, RGB
                        if len(image_bytes) == 3072:
                            yield {
                                "images": np.array(
                                    Image.frombytes(
                                        mode="RGB",
                                        data=image_bytes,
                                        size=(32, 32),
                                    ),
                                    dtype=np.uint8,
                                ).reshape(-1, 3),
                                "captions": caption_row[0],
                            }


def create_cc_dataset(
    train_report_file: str,
    val_report_file: str,
    train_caption_file: str,
    val_caption_file: str,
    output_path: str,
):
    """Create Conceptual Captions dataset."""
    features = Features(
        {
            "captions": Value(dtype="string"),
            "images": Array2D(shape=(1024, 3), dtype="uint8"),
        }
    )

    train_dataset = Dataset.from_generator(
        create_cc_dataset_split,
        features=features,
        gen_kwargs={
            "report_file": train_report_file,
            "captions_file": train_caption_file,
            "split": "train",
        },
    )
    validation_dataset = Dataset.from_generator(
        create_cc_dataset_split,
        features=features,
        gen_kwargs={
            "report_file": val_report_file,
            "captions_file": val_caption_file,
            "split": "validation",
        },
    )

    full_dataset = DatasetDict()
    full_dataset["train"] = train_dataset
    full_dataset["validation"] = validation_dataset

    full_dataset.save_to_disk(output_path)
    # full_dataset.push_to_hub("h4rr9/conceptual_captions_32x32", private=True)


if __name__ == "__main__":
    create_cc_dataset(
        "./data/cc/downloaded_training_report.tsv",
        "./data/cc/downloaded_validation_report.tsv",
        "./data/cc/Train_GCC-training.tsv",
        "./data/cc/Validation_GCC-1.1.0-Validation.tsv",
        "./data/cc/dataset",
    )

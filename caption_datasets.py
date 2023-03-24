#!/usr/bin/env python3

from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    def __init__(self):
        raise NotImplementedError("Dataset base not implemented yet")


class ConceptualCaptions(ImageCaptionDataset):
    def __init__(self):
        raise NotImplementedError("Conceptual Captions dataset not implemented yet")


class ImageNet(ImageCaptionDataset):
    def __init__(self):
        raise NotImplementedError("ImageNet dataset not implemented yet")


class MSCOCO(ImageCaptionDataset):
    def __init__(self):
        raise NotImplementedError("MSCOCO 2017 dataset not implemented yet")

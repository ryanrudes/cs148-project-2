from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
import os

from augmentations import get_preprocessor
from data_utils import load_data


def parse_args():
    parser = ArgumentParser(description="Preprocess the dataset and compute mean/std.")
    parser.add_argument("--color", action="store_true", help="Whether to preprocess images in color (RGB) or grayscale.")
    parser.add_argument("--size", type=int, default=64, help="Size to resize images to (default: 64).")
    parser.add_argument("--id", type=str, help="ID for the preprocessed dataset (e.g. mnist_rgb_64).")
    return parser.parse_args()


def load_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), "data/dataset")
    data: list[dict] = load_data(data_dir)

    images = []
    labels = []

    for item in data:
        images.append(item['img'])
        labels.append(item['label'])

    return images, labels


def print_dataset_frequencies(images: list, labels: list):
    print("Number of images:", len(images))
    print("Number of labels:", len(labels))

    label_counts = Counter(labels)
    print("Label distribution:")
    for digit in range(10):
        print(f"Digit {digit}: {label_counts[digit]} samples")


def preprocess_images(images: list, color: bool, size: int):
    # Build and run the deterministic preprocessor, which includes resizing and color conversion
    preprocess = get_preprocessor(color, size)

    preprocessed_images = []
    for img in tqdm(images, desc="Preprocessing images"):
        x = preprocess(img)
        preprocessed_images.append(x.numpy())

    return preprocessed_images


def compute_mean_std(images: list, color: bool):
    num_channels = 3 if color else 1
    s1 = torch.zeros(num_channels)
    s2 = torch.zeros(num_channels)
    n = 0

    for x in tqdm(images, desc="Computing mean/std"):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        s1 += x.sum((1, 2))
        s2 += (x * x).sum((1, 2))
        n += x.shape[1] * x.shape[2]

    mean_t = s1 / n
    std_t = torch.sqrt(s2 / n - mean_t ** 2)
    mean = tuple(mean_t.tolist())
    std = tuple(std_t.tolist())

    return mean, std


if __name__ == "__main__":
    args = parse_args()

    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", f"{args.id}.npz")

    if os.path.exists(dataset_path):
        print(f"Dataset '{args.id}' already exists. Are you sure you want to overwrite it? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            exit()

    images, labels = load_dataset()
    print_dataset_frequencies(images, labels)

    preprocessed_images = preprocess_images(images, args.color, args.size)
    mean, std = compute_mean_std(preprocessed_images, args.color)
    print("Computed mean:", mean)
    print("Computed std:", std)

    preprocessed_images = np.array(preprocessed_images)
    print("Preprocessed images shape:", preprocessed_images.shape)

    dataset = {"images": preprocessed_images, "labels": labels, "mean": mean, "std": std}
    np.savez_compressed(dataset_path, **dataset)

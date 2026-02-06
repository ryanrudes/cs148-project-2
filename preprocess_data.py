from collections import Counter
from tqdm import tqdm

import numpy as np
import os

from data_utils import load_data
from augmentations import preprocess

data_dir = os.path.join(os.path.dirname(__file__), "data/dataset")
data: list[dict] = load_data(data_dir)

images = []
labels = []

for item in data:
    images.append(item['img'])
    labels.append(item['label'])

print("Number of images:", len(images))
print("Number of labels:", len(labels))

label_counts = Counter(labels)
print("Label distribution:")
for digit in range(10):
    print(f"Digit {digit}: {label_counts[digit]} samples")

# Do all of the deterministic preprocessing first, then apply augmentation on the fly in the dataset
# during training
preprocessed_images = np.array([preprocess(img) for img in tqdm(images, desc="Preprocessing images")])
print("Preprocessed images shape:", preprocessed_images.shape)

dataset = {"images": preprocessed_images, "labels": labels}
np.savez_compressed("preprocessed_data_color.npz", **dataset)

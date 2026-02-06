from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DummyMNIST(Dataset):
    def __init__(
        self,
        images: list[Image.Image],
        labels: list[int],
        transform: transforms.Compose = None
    ):
        """
        Dummy MNIST dataset class.

        Args:
            images: list of PIL.Image objects
            labels: list of ints representing the digit in each image
            transform: optional torchvision transform
        """

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

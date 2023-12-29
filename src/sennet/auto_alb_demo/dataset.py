from sennet.environments.constants import DATA_DIR
import cv2
import torchvision
from pathlib import Path

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root: str | None | Path = None, train=True, download=True, transform=None):
        if root is None:
            root = str(DATA_DIR / "cifar10")
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
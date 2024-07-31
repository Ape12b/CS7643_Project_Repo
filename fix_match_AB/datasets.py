from torchvision import datasets, transforms
from randaugment import RandAugmentMC
from PIL import Image
import numpy as np

class TransformFix:
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class get_cifar(datasets.CIFAR10):
    def __init__(self, root, indexes, transform=None):
        super().__init__(root, train=True, transform=transform, download=True)
        self.data = self.data[indexes]
        self.targets = np.array(self.targets)[indexes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


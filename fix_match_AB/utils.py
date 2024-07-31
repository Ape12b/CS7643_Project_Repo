import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
from datasets import get_cifar, TransformFix
from sklearn.model_selection import train_test_split


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., np.cos(np.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_cifar10(root, num_labeled, num_expand_x, num_expand_u):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(base_dataset.targets, num_labeled, num_expand_x, num_expand_u, num_classes=10, total_size=len(base_dataset))

    train_labeled_dataset = get_cifar(root, train_labeled_idxs, transform=transform_labeled)
    train_unlabeled_dataset = get_cifar(root, train_unlabeled_idxs, transform=TransformFix(mean=cifar10_mean, std=cifar10_std))
    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split(labels, num_labeled, num_expand_x, num_expand_u, num_classes, total_size):
    unlabeled_ratio = 1 - num_labeled / total_size
    labeled_idx, unlabeled_idx = train_test_split(np.arange(total_size), test_size=unlabeled_ratio, shuffle=True, stratify=labels)
    labeled_idx = np.tile(labeled_idx, num_expand_x // len(labeled_idx) + 1)[:num_expand_x]
    unlabeled_idx = np.tile(unlabeled_idx, num_expand_u // len(unlabeled_idx) + 1)[:num_expand_u]
    return labeled_idx, unlabeled_idx
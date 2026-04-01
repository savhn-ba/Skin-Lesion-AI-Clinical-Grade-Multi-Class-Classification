"""Image transforms for HAM10000 train and validation splits."""

from torchvision import transforms

# ImageNet statistics — used because EfficientNet-B3 was pre-trained on ImageNet.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# EfficientNet-B3 expects 300 × 300 inputs.
_INPUT_SIZE = 300


def get_train_transforms() -> transforms.Compose:
    """Return augmentation pipeline for training images.

    Applies random flips, rotations, colour jitter, and random erasing to
    improve generalisation and simulate the natural variation of dermoscopy
    images acquired with different devices and lighting conditions.
    """
    return transforms.Compose(
        [
            transforms.Resize((_INPUT_SIZE + 32, _INPUT_SIZE + 32)),
            transforms.RandomCrop(_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            # Randomly erase a rectangular patch to reduce reliance on single
            # features (acts as a regulariser similar to cutout).
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Return deterministic transforms for validation / test images."""
    return transforms.Compose(
        [
            transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )

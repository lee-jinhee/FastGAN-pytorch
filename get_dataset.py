from torch.utils.data import Dataset
from torchvision.datasets.celeba import CelebA
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms

def get_cifar10_transform():
    img_size = 32
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    return transform

def get_celeba_transform():
    img_size = 64
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    return transform

TRANSFORM_DICT = {
    'celeba': get_celeba_transform,
    'cifar10': get_cifar10_transform,
}

DATASET_DICT = {
    'cifar10': CIFAR10,
    'celeba': CelebA,
}

def get_transform(dataset_name):
    transform = TRANSFORM_DICT[dataset_name]()
    return transform

def get_predefined_dataset(dataset_name, root, **kwargs):
    transform = get_transform(dataset_name)
    dataset = DATASET_DICT[dataset_name](root=root, transform=transform, download=True, **kwargs)
    return dataset
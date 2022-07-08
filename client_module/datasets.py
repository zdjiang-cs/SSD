import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from randaugment import RandAugmentMC


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.classes = data.classes

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataLoaderHelper(object):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.dataiter = iter(self.loader)

    def __next__(self):
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            data, target = next(self.dataiter)
        
        return data, target

class RandomPartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in partition_sizes:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        # sizes is a class_num * vm_num matrix
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        for class_idx in range(len(data.classes)):
            label_indexes.append(list(np.where(np.array(data.targets) == class_idx)[0]))
            class_len.append(len(label_indexes[class_idx]))
            rng.shuffle(label_indexes[class_idx])
        
        # distribute class indexes to each vm according to sizes matrix
        for class_idx in range(len(data.classes)):
            begin_idx = 0
            for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                end_idx = begin_idx + round(frac * class_len[class_idx])
                self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True):
    if selected_idxs is None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = DataLoader(partition, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory)
    
    return DataLoaderHelper(dataloader)

def load_datasets(dataset_type, data_path="data_path"):
    train_transform = load_default_transform(dataset_type, train=True)
    s_train_transform = load_default_transform(dataset_type, train=True, strong_augment=True)
    test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
        s_train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=s_train_transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(data_path, train = True, 
                                            download = True, transform=train_transform)
        s_train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=s_train_transform)
        test_dataset = datasets.FashionMNIST(data_path, train = False, 
                                            download = True, transform=test_transform)

    return train_dataset, s_train_dataset, test_dataset


def load_default_transform(dataset_type, train=False, strong_augment = False):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        if train:
            dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

        if strong_augment:
            dataset_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(size=32,
                                                    padding=int(32*0.125),
                                                    padding_mode='reflect'),
                            RandAugmentMC(n=2, m=10),
                            transforms.ToTensor(),
                            normalize
                        ])

    elif dataset_type == 'FashionMNIST':
        dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
        if strong_augment:
            dataset_transform = transforms.Compose([
                            RandAugmentMC(n = 2, m = 10),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])
    

    return dataset_transform

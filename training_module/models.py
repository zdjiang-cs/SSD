import torch
import torch.nn.functional as F
import torch.nn as nn


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 64)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def create_model_instance(dataset_type, model_type):
    if dataset_type == 'MNIST' or 'FashionMNIST':
        if model_type == 'CNN':
            model = MNIST_Net()

    return model
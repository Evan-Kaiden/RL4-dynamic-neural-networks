import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transforms.ToTensor())
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

batch_size = 1
pretrainloader = DataLoader(train_ds, batch_size=32, shuffle=True)
trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

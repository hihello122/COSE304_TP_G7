# data/mnist.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, test_loader

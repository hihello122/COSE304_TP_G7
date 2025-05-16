# main.py

import argparse
import yaml
import torch

import argparse
import yaml
import torch

from data import mnist, cifar
from models.flow import ResidualFlow
from train.trainer import MemoryEfficientTrainer

def parse_args():
    # Use this function to get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient training')

    #Todo: add more arguments as needed
    #parser.add_argument('--learninglate', type=int, required=True, help='learning rate of learning', default=0.001)

    return parser.parse_args()

def load_config(path):
    # not completed made the confignuration file
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if config['data']['name'] == 'mnist':
        train_loader, test_loader = mnist.get_dataloaders(config['data'])
        input_shape = (1, 28, 28) # MNIST 입력 형태
    elif config['data']['name'] == 'cifar':
        train_loader, test_loader = cifar.get_dataloaders(config['data'])
        input_shape = (3, 32, 32) # CIFAR 입력 형태
    else:
        raise ValueError(f"Unknown dataset: {config['data']['name']}")

    # Build model
    model = ResidualFlow(config['model']).to(device)

    # Set up trainer based on the --memory_efficient flag
    if args.memory_efficient:
        trainer = MemoryEfficientTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            input_shape=input_shape # 입력 형태 전달
        )
        print("Using Memory-Efficient Trainer")
    else:
        trainer = MemoryEfficientTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=device,
        )
        print("Using Standard Trainer")

    # Start training
    trainer.train_epoch(train_loader)

if __name__ == '__main__':
    main()
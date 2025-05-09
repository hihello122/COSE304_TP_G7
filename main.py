# main.py

import argparse
import yaml
import torch

from data import mnist, cifar
from models.flow import ResidualFlow
from train.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    return parser.parse_args()

def load_config(path):
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
    elif config['data']['name'] == 'cifar':
        train_loader, test_loader = cifar.get_dataloaders(config['data'])
    else:
        raise ValueError(f"Unknown dataset: {config['data']['name']}")

    # Build model
    model = ResidualFlow(config['model']).to(device)

    # Set up trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Start training
    trainer.train()

if __name__ == '__main__':
    main()

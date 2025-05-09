# train/trainer.py

import torch
import torch.nn.functional as F
import os

class Trainer:
    def __init__(self, model, train_loader, test_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    def train(self):
        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.to(self.device)
                x = x.view(x.size(0), -1)  # flatten

                z, log_det = self.model(x)
                log_prob = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * 3.1416))).sum(dim=1)
                loss = -(log_prob + log_det).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.train_loader):.4f}")

            # 모델 저장
            os.makedirs(self.config['training']['save_path'], exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.config['training']['save_path'], f"model_epoch_{epoch+1}.pt"))

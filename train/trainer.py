# train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MemoryEfficientTrainer:
    def __init__(self, model, train_loader, test_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.input_shape = config['training']['input_shape']  # Assuming input shape is passed in the config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    def neumann_gradient_estimator(self, Jg, v, N):
        """
        Neumann series based unbiased estimator for the gradient of log det(I + Jg).

        Args:
            Jg (torch.Tensor): Jacobian matrix of g(x, theta).
            v (torch.Tensor): Random vector sampled from N(0, I).
            N (int): Number of terms to use in the Neumann series.

        Returns:
            torch.Tensor: Unbiased estimate of the gradient.
        """
        batch_size = Jg.shape[0]
        m = Jg.shape[-1]
        grad_est = torch.zeros_like(next(self.model.parameters())).float().to(self.device)

        for k in range(N):
            # v를 (batch_size, m, 1) 형태로 reshape
            v_reshaped = v.unsqueeze(-1)
            Jg_k_v = v_reshaped  # (batch_size, m, 1)
            for _ in range(k):
                Jg_k_v = torch.bmm(Jg, Jg_k_v)  # (batch_size, m, 1)

            # v를 (batch_size, 1, m) 형태로 reshape하여 matmul 수행 후 squeeze
            term = torch.bmm(v.unsqueeze(1), Jg_k_v).squeeze(-1).squeeze(-1)  # (batch_size,)

            # Need to compute the gradient of Jg w.r.t. parameters
            grad_Jg = torch.autograd.grad(Jg, self.model.parameters(),
                                         grad_outputs=torch.eye(m).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device),
                                         create_graph=True, allow_unused=True) # allow_unused 추가

            grad_term = torch.zeros_like(grad_est).float().to(self.device)
            param_count = 0
            for i, param in enumerate(self.model.parameters()):
                n_elements = param.numel()
                grad_Jg_param = grad_Jg[i]
                if grad_Jg_param is not None:
                    # term (batch_size,), grad_Jg_param (batch_size, out_dim, in_dim) -> (batch_size, in_dim)
                    # param (in_dim, out_dim) or (out_dim,) or ... 다양한 형태 고려 필요
                    if param.ndim == 2: # Linear layer weight (out_dim, in_dim)
                        grad_term_param = torch.einsum('b,bo->bo', term, grad_Jg_param) # (batch_size, out_dim)
                        grad_term[param_count:param_count + n_elements] += grad_term_param.reshape(-1)
                    elif param.ndim == 1: # Bias (out_dim,)
                        grad_term[param_count:param_count + n_elements] += torch.sum(torch.einsum('b,b->b', term, grad_Jg_param), dim=0)
                    elif grad_Jg_param.ndim >= 2:
                        # 일반적인 텐서 곱셈 (shape에 따라 조정 필요)
                        grad_term_param = torch.einsum('b,...o->b...o', term, grad_Jg_param)
                        grad_term[param_count:param_count + n_elements] += grad_term_param.reshape(-1)
                param_count += n_elements

            prob_Nk_ge_k = 1.0  # Assuming uniform distribution for N for simplicity
            grad_est += ((-1)**k * prob_Nk_ge_k * grad_term) / N

        return grad_est

    def backward_in_forward(self, x, target):
        """
        Performs a forward pass and computes gradients using standard backpropagation.

        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Loss value.
        """
        self.optimizer.zero_grad()
        output = self.model(x)
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, dataloader, method='standard', n_neumann_terms=10):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            if method == 'standard':
                loss = self.backward_in_forward(data, target)
            elif method == 'neumann':
                # Jacobian 계산 시 입력 형태에 맞춰야 함. g(x, theta)의 출력이 x와 같은 형태라고 가정
                def compute_jacobian(outputs, inputs):
                    jacobian = []
                    flat_outputs = outputs.view(outputs.size(0), -1)
                    grad_outputs = torch.eye(flat_outputs.size(-1)).to(inputs.device)
                    for i in range(flat_outputs.size(-1)):
                        grad_input = torch.autograd.grad(flat_outputs[:, i], inputs, grad_outputs=grad_outputs[:, i], retain_graph=True, allow_unused=True)[0]
                        jacobian.append(grad_input.view(*outputs.shape))
                    return torch.stack(jacobian, dim=1) # (batch_size, output_dim, *input_shape)

                self.model.zero_grad()
                output = self.model(data) # g(x, theta) 역할을 한다고 가정
                Jg = compute_jacobian(output, data) # data에 대한 Jacobian 계산

                batch_size, out_dim, *input_dims = Jg.shape
                v = torch.randn(batch_size, out_dim).to(self.device)
                N = np.random.randint(1, n_neumann_terms + 1)

                grad_logdet = self.neumann_gradient_estimator(Jg.view(batch_size, out_dim, -1), v, N) # Jacobian reshape 필요
                # 여기서 grad_logdet을 사용하여 optimizer 업데이트를 수행해야 함
                # (loss function에 log det term이 포함되어 있다고 가정)
                loss = torch.tensor(0.0).to(self.device) # 예시, 실제 loss 계산 필요

                self.optimizer.zero_grad() # optimizer 초기화 다시 필요
                # grad_logdet을 각 parameter의 .grad에 할당
                param_count = 0
                for param in self.model.parameters():
                    n_elements = param.numel()
                    param.grad = grad_logdet[param_count:param_count + n_elements].reshape(param.shape)
                    param_count += n_elements
                self.optimizer.step()

            else:
                raise ValueError(f"Unknown training method: {method}")
            total_loss += loss.item() * len(data)
        return total_loss / len(dataloader.dataset)


# Example Usage:
if __name__ == '__main__':
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Hyperparameters
    input_size = 784
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    batch_size = 64
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy dataset and dataloader
    X = torch.randn(1000, 1, 28, 28)
    y = torch.randint(0, output_size, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Initialize model, optimizer, and trainer
    model = SimpleModel(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = MemoryEfficientTrainer(model, optimizer, device)

    # Train using standard backpropagation (backward-in-forward in this context)
    print("Training with standard backpropagation:")
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(dataloader, method='standard')
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")

    # To train using the Neumann gradient estimator, you would need a loss function
    # that incorporates the log determinant term and a way to compute the Jacobian Jg
    # based on your model's architecture. The 'train_step_neumann' function provides
    # a template for how you might approach this.

    # Example of how you might call the Neumann training (requires defining Jg):
    # print("\nTraining with Neumann Gradient Estimator:")
    # for epoch in range(num_epochs):
    #     train_loss = trainer.train_epoch(dataloader, method='neumann', n_neumann_terms=5)
    #     print(f"Epoch {epoch+1}, Loss (Neumann): {train_loss:.4f}")

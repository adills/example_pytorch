import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def to_tensor(array, dtype=torch.float32):
    """
    Converts input to a PyTorch tensor if it's not already a tensor.
    """
    if isinstance(array, torch.Tensor):
        return array
    return torch.tensor(array, dtype=dtype)

# Truth solution
def solution(t):
    t = np.asarray(t)  # Ensure it's a standard ndarray
    x_t = 0.5 * (np.exp(-t) + np.exp(-3*t))
    y_t = 0.5 * (-1*np.exp(-t) + np.exp(-3*t))
    return x_t, y_t

def loss_data(data, pred):
    """
    Data Loss Function: Computes the mean squared error (MSE)
    between data and predictions across all dimensions.
    
    Parameters:
    - data (tuple of torch.Tensor): Ground truth values (e.g., (x_true, y_true)).
    - pred (tuple of torch.Tensor): Predicted values (e.g., (x_pred, y_pred)).

    Returns:
    - torch.Tensor: The computed MSE loss.
    """
    # Unpack tuples
    data_x, data_y = data
    pred_x, pred_y = pred

    # Compute MSE separately for both components and sum
    loss_x = torch.mean((data_x - pred_x) ** 2)
    loss_y = torch.mean((data_y - pred_y) ** 2)

    return loss_x + loss_y

class PINN(nn.Module):
    def __init__(self, hidden_size=64, output_size=2):
        super(PINN, self).__init__()
        # Input layer
        self.in_layer = nn.Linear(1, hidden_size)
        # Hidden layers
        self.h1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)  # Adding dropout for regularization
        )
        # Output layer
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, t):
        x = torch.sin(t)  # Assuming t is in radians
        x = self.in_layer(x)
        x = self.h1(x)
        pred = self.out_layer(x)
        return pred

class PINNDataset(Dataset):
    def __init__(self, t, x, y):
        self.x = to_tensor(x)#torch.FloatTensor(x)
        self.y = to_tensor(y)#torch.FloatTensor(y)
        self.t = to_tensor(t)#torch.FloatTensor(t)
        # print(f'PINNDataset.x.shape = {x.shape}')

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return (self.t[idx], self.x[idx], self.y[idx])
    

class PINNTrainer:
    def __init__(self, model, optimizer, train_loader, test_loader):
        self.model = model.to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        # print(f'len(train_loader.dataset) = {len(train_loader.dataset)}')
        # print(f'train_loader.dataset[0] = {train_loader.dataset[0]}')
        # print(f'train_loader.dataset[1] = {train_loader.dataset[1]}')
        # print(f'train_loader.dataset[2] = {train_loader.dataset[2]}')

    def train(self, num_epochs=4000):
        for epoch in range(num_epochs):
            self.model.zero_grad()
            
            batch_t, batch_x, batch_y = next(iter(self.train_loader))
            batch_t = batch_t.clone().detach().requires_grad_(True)
            print(f"Batch size: {len(batch_x)}")
            # Forward pass and compute loss
            output = self.model(batch_t)
            x_pred, y_pred = output.chunk(2, dim=-1)
            
            # Compute residuals
            # dxdt = torch.autograd.grad(x_pred, batch_t, create_graph=True)[0]
            # dydt = torch.autograd.grad(y_pred, batch_t, create_graph=True)[0]
            dxdt = torch.autograd.grad(x_pred, batch_t, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
            dydt = torch.autograd.grad(y_pred, batch_t, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
            
            res_x = dxdt + 2 * x_pred + y_pred
            res_y = dydt + x_pred + 2 * y_pred
            
            loss_residuals = (res_x ** 2).mean() + (res_y ** 2).mean()
            print(f"Loss: residual = {loss_residuals:6f}")

            # Initial Conditions Loss
            ic_loss = (x_pred[0] - 1) ** 2 + (y_pred[0] - 0) ** 2
            print(f"Loss: IC = {ic_loss.item():6f}")
            
            # Data Loss
            data_loss = loss_data((batch_x, batch_y), (x_pred, y_pred))
            print(f"Loss: Data = {data_loss.item():6f}")
            
            total_loss = loss_residuals + ic_loss + data_loss
            
            # Backpropagation and optimization
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch} Loss: {total_loss.item():6f}")
    
    def test(self):
        predictions = []
        for t, x_test_orig, y_test_orig in self.test_loader:
            with torch.no_grad():
                output = self.model(t)
                analytical = solution(t)
                x_test_sol, y_test_sol = analytical[:,0], analytical[:,1]
                # analytical = torch.stack([to_tensor(analytical[:,0]), to_tensor(analytical[:,1])], dim=-1)
                x_pred, y_pred = output.chunk(2, dim=-1)
                # convert a PyTorch tensor to a NumPy array with .cpu().numpy()
                predictions.append((x_pred.cpu().numpy(), 
                                    y_pred.cpu().numpy(), 
                                    x_test_sol, 
                                    y_test_sol))
        
        return predictions
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
import copy
import ctypes
try:
    ctypes.CDLL("/home/marfred/.conda/envs/cadmm/lib/libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Use GPU 3 as specified, or fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Neural Network Architecture as specified in the paper
class DiNNOMNISTNet(nn.Module):
    def __init__(self):
        super(DiNNOMNISTNet, self).__init__()
        # "A convolutional layer with three 5x5 filters"
        self.conv = nn.Conv2d(1, 3, kernel_size=5) 
        # 28x28 -> conv 5x5 -> 24x24. Flattened size = 3 * 24 * 24 = 1728
        # "followed by 2 linear layers of width 576, 64"
        self.fc1 = nn.Linear(1728, 576)
        self.fc2 = nn.Linear(576, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # "and a log-softmax output layer"
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 2. Node definition representing one robot/agent
class DiNNONode:
    def __init__(self, node_id, model, dataloader, lr=1e-3):
        self.node_id = node_id
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = dataloader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Count total parameters for loss normalization
        self.num_params = sum(p.numel() for p in self.model.parameters())
        
        # Dual variables p_i and stored neighbor weights moved to device
        self.p = [torch.zeros_like(param).to(device) for param in self.model.parameters()]
        self.theta_k = [param.clone().detach().to(device) for param in self.model.parameters()]
        self.neighbors_theta_k = {} # Will store neighboring weights from communication

    def update_dual_variables(self, rho):
        # p_i^{k+1} = p_i^k + \rho \sum_{j \in N_i} (\theta_i^k - \theta_j^k)
        for j, neighbor_theta in self.neighbors_theta_k.items():
            for p_tensor, my_theta, their_theta in zip(self.p, self.theta_k, neighbor_theta):
                p_tensor.add_(rho * (my_theta - their_theta))

    def approximate_primal_update(self, rho, B_steps):
            self.model.train()
            data_iter = iter(self.dataloader)
            total_task_loss = 0.0
            total_penalty_loss = 0.0
            
            for tau in range(B_steps):
                try:
                    data, target = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    data, target = next(data_iter)
                
                data, target = data.to(device), target.to(device)
                    
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # 1. Standard task loss (Negative Log-Likelihood)
                task_loss = F.nll_loss(output, target)
                
                # 2. Add DiNNO Augmented Lagrangian penalties (Normalized by num_params)
                dual_term = 0.0
                penalty_term = 0.0
                
                for param_idx, (param, p_tensor, my_theta_k) in enumerate(zip(self.model.parameters(), self.p, self.theta_k)):
                    # \theta^T p_i^{k+1}
                    dual_term += torch.sum(param * p_tensor)
                    
                    # \rho \sum_{j \in N_i} || \theta - (\theta_i^k + \theta_j^k)/2 ||^2_2
                    for j, neighbor_theta in self.neighbors_theta_k.items():
                        target_center = (my_theta_k + neighbor_theta[param_idx]) / 2.0
                        penalty_term += rho * torch.sum((param - target_center) ** 2)

                # Scale the entire augmented Lagrangian component by num_params
                consensus_loss = (dual_term + penalty_term) / self.num_params
                total_loss = task_loss + consensus_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                total_task_loss += task_loss.item()
                total_penalty_loss += consensus_loss.item()
                
            # Update our stored state for the next communication round
            self.theta_k = [param.clone().detach() for param in self.model.parameters()]
            return total_task_loss / B_steps, total_penalty_loss / B_steps

# 3. Main Training Loop
def train_dinno():
    # Setup Heterogeneous Data: Node 0 gets digits 0-4, Node 1 gets digits 5-9
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    
    # Use torch.as_tensor to avoid DeprecationWarning and handle GPU if needed
    targets = torch.as_tensor(dataset.targets)
    idx_node0 = torch.where(targets < 5)[0]
    idx_node1 = torch.where(targets >= 5)[0]
    
    loader0 = DataLoader(Subset(dataset, idx_node0), batch_size=64, shuffle=True)
    loader1 = DataLoader(Subset(dataset, idx_node1), batch_size=64, shuffle=True)

    # Initialize nodes with a shared initial weight configuration
    initial_model = DiNNOMNISTNet()
    node0 = DiNNONode(node_id=0, model=initial_model, dataloader=loader0)
    node1 = DiNNONode(node_id=1, model=initial_model, dataloader=loader1)
    nodes = [node0, node1]

    # Hyperparameters
    COMMUNICATION_ROUNDS = 1000
    B_STEPS = 10     # Reduced local steps for better consensus
    RHO = 5.0       # Increased penalty parameter for stronger consensus

    for k in range(COMMUNICATION_ROUNDS):
        # Step 1: Communication (Simulate graph message passing)
        node0.neighbors_theta_k[1] = [p.clone().detach() for p in node1.theta_k]
        node1.neighbors_theta_k[0] = [p.clone().detach() for p in node0.theta_k]
        
        # Step 2 & 3: DiNNO updates
        stats = []
        for node in nodes:
            # Update dual variable using newly received neighbor weights
            node.update_dual_variables(RHO)
            
            # Perform approximate primal update (B steps of SGD/Adam)
            t_loss, p_loss = node.approximate_primal_update(RHO, B_STEPS)
            stats.append((t_loss, p_loss))
            
        # Evaluate consensus and loss periodically
        if k % 10 == 0:
            avg_task_loss = sum(s[0] for s in stats) / len(stats)
            avg_pen_loss = sum(s[1] for s in stats) / len(stats)
            # Measure weight discrepancy (consensus error)
            diff = sum(torch.sum((p0 - p1)**2) for p0, p1 in zip(node0.theta_k, node1.theta_k))
            print(f"Round {k:3d} | Task Loss: {avg_task_loss:.4f} | Pen Loss: {avg_pen_loss:.4f} | Consensus Error: {diff.item():.4f}")

if __name__ == "__main__":
    train_dinno()
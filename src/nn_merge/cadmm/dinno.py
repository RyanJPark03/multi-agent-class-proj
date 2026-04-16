import torch

class DiNNOManager:
    """
    Manages DiNNO (Distributed Neural Network Optimization) state for a single node.
    It tracks dual variables and neighbor parameter snapshots, and computes the
    augmented Lagrangian penalty to add to the local task loss.
    """
    def __init__(self, parameters, node_id=None):
        """
        Args:
            parameters: Iterable of torch.Tensor (e.g., from model.parameters())
            node_id: Optional identifier for this node
        """
        self.node_id = node_id
        self.parameters = list(parameters)
        
        # Calculate total parameters for loss normalization
        self.num_params = sum(p.numel() for p in self.parameters)
        
        # Determine device from the first parameter (default to CPU if empty)
        self.device = self.parameters[0].device if self.parameters else torch.device("cpu")
        
        # Initialize dual variables and local snapshot on the same device
        self.p = [torch.zeros_like(param).to(self.device) for param in self.parameters]
        self.theta_k = [param.clone().detach().to(self.device) for param in self.parameters]
        
        # Dictionary mapping neighbor_id -> list of tensor snapshots
        self.neighbors_theta_k = {}

    def to(self, device):
        """Moves internal state to the specified device."""
        self.device = device
        self.p = [p.to(device) for p in self.p]
        self.theta_k = [t.to(device) for t in self.theta_k]
        for v in self.neighbors_theta_k.values():
            for i in range(len(v)):
                v[i] = v[i].to(device)
        return self

    def update_snapshot(self):
        """Snapshots the current model parameters into theta_k to be shared with neighbors."""
        self.theta_k = [param.clone().detach().to(self.device) for param in self.parameters]

    def receive_neighbor_snapshot(self, neighbor_id, neighbor_theta_k):
        """
        Stores a snapshot of a neighbor's parameters.
        Args:
            neighbor_id: Identifier for the neighbor.
            neighbor_theta_k: List of parameter tensors from the neighbor.
        """
        self.neighbors_theta_k[neighbor_id] = [t.clone().detach().to(self.device) for t in neighbor_theta_k]

    def update_dual_variables(self, rho):
        """
        Updates the dual variables for this node based on the consensus difference.
        p_i^{k+1} = p_i^k + \rho \sum_{j \in N_i} (\theta_i^k - \theta_j^k)
        """
        for neighbor_theta in self.neighbors_theta_k.values():
            for p_tensor, my_theta, their_theta in zip(self.p, self.theta_k, neighbor_theta):
                p_tensor.add_(rho * (my_theta - their_theta))

    def compute_consensus_loss(self, rho):
        """
        Computes the DiNNO Augmented Lagrangian consensus term.
        This scalar loss should be added to the standard task loss before calling .backward().
        
        Returns:
            torch.Tensor: The computed consensus penalty loss term scalar (differentiable w.r.t parameters).
        """
        dual_term = 0.0
        penalty_term = 0.0
        
        # We process parameter updates keeping gradients flowing through self.parameters only
        for param_idx, (param, p_tensor, my_theta_k) in enumerate(zip(self.parameters, self.p, self.theta_k)):
            # \theta^T p_i^{k+1}
            dual_term += torch.sum(param * p_tensor)
            
            # \rho \sum_{j \in N_i} || \theta - (\theta_i^k + \theta_j^k)/2 ||^2_2
            for neighbor_theta in self.neighbors_theta_k.values():
                target_center = (my_theta_k + neighbor_theta[param_idx]) / 2.0
                penalty_term += rho * torch.sum((param - target_center) ** 2)

        # Scale the entire augmented Lagrangian component by num_params for stability
        consensus_loss = (dual_term + penalty_term) / self.num_params
        return consensus_loss

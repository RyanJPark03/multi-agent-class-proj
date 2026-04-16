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
        self.rho = 1.0 # Default rho

    def set_rho(self, rho):
        self.rho = rho

    def get_consensus_grad(self, param_idx):
        """
        Calculates the consensus gradient term for a specific parameter:
        grad = p_i + rho * sum_{j in N_i} (theta - (theta_i + theta_j)/2)
        """
        p_tensor = self.p[param_idx]
        my_theta_k = self.theta_k[param_idx]
        param = self.parameters[param_idx]
        
        grad_term = torch.zeros_like(param)
        # \theta^T p_i part gradient is just p_i
        grad_term.add_(p_tensor)
        
        # \rho \sum_{j \in N_i} || \theta - (\theta_i^k + \theta_j^k)/2 ||^2_2 part gradient
        for neighbor_theta in self.neighbors_theta_k.values():
            target_center = (my_theta_k + neighbor_theta[param_idx]) / 2.0
            # Derivative of rho * (x - c)^2 is 2 * rho * (x - c)
            grad_term.add_(2.0 * self.rho * (param - target_center))
            
        return grad_term / self.num_params

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
        r"""
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

from stable_baselines3.common.callbacks import BaseCallback

class DiNNOCallback(BaseCallback):
    """
    Stable Baselines 3 Callback to integrate DiNNO/CADMM consensus.
    Registers backward hooks on policy parameters to inject consensus gradients.
    """
    def __init__(self, node_id, rho=1.0, registry=None, communication_freq=1000, verbose=0):
        super().__init__(verbose)
        self.node_id = node_id
        self.rho = rho
        self.registry = registry # Shared dictionary for snapshots
        self.communication_freq = communication_freq
        self.dinno_manager = None
        self.hooks = []

    def _init_callback(self):
        # In BaseCallback, self.model is already assigned before this call
        # Initialize manager with policy parameters
        params = list(self.model.policy.parameters())
        self.dinno_manager = DiNNOManager(params, node_id=self.node_id)
        self.dinno_manager.set_rho(self.rho)
        self.dinno_manager.to(self.model.device)

        # Register backward hooks
        for i, param in enumerate(params):
            if param.requires_grad:
                hook = self._make_hook(i)
                self.hooks.append(param.register_hook(hook))

    def _make_hook(self, idx):
        def hook(grad):
            return grad + self.dinno_manager.get_consensus_grad(idx)
        return hook

    def _on_rollout_end(self) -> bool:
        """Triggered at the end of every rollout (PPO)."""
        self._sync()
        return True

    def _on_step(self) -> bool:
        """Triggered every step (SAC)."""
        if self.model.num_timesteps % self.communication_freq == 0:
            self._sync()
        return True

    def _sync(self):
        """Update snapshot, exchange with neighbors, and update dual variables."""
        if self.dinno_manager is None:
            return
            
        # 1. Update own snapshot
        self.dinno_manager.update_snapshot()
        
        # 2. Post to registry
        if self.registry is not None:
            self.registry[self.node_id] = self.dinno_manager.theta_k
            
            # 3. Read neighbors from registry (Assume all other entries are neighbors)
            for other_id, snapshot in self.registry.items():
                if other_id != self.node_id:
                    self.dinno_manager.receive_neighbor_snapshot(other_id, snapshot)
        
        # 4. Update dual variables
        self.dinno_manager.update_dual_variables(self.rho)


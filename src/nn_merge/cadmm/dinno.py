import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class DiNNOManager:
    """
    Manages DiNNO (Distributed Neural Network Optimization) state for a single node.
    It tracks dual variables and neighbor parameter snapshots, and computes the
    augmented Lagrangian penalty to add to the local task loss.
    """
    def __init__(self, parameters, node_id=None):
        self.node_id = node_id
        self.parameters = list(parameters)
        
        self.num_params = sum(p.numel() for p in self.parameters)
        self.device = self.parameters[0].device if self.parameters else torch.device("cpu")
        
        self.p = [torch.zeros_like(param).to(self.device) for param in self.parameters]
        self.theta_k = [param.clone().detach().to(self.device) for param in self.parameters]
        
        self.neighbors_theta_k = {}
        self.rho = 1.0 

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
        grad_term.add_(p_tensor)
        
        for neighbor_theta in self.neighbors_theta_k.values():
            target_center = (my_theta_k + neighbor_theta[param_idx]) / 2.0
            grad_term.add_(2.0 * self.rho * (param - target_center))
            
        return grad_term

    def to(self, device):
        self.device = device
        self.p = [p.to(device) for p in self.p]
        self.theta_k = [t.to(device) for t in self.theta_k]
        for v in self.neighbors_theta_k.values():
            for i in range(len(v)):
                v[i] = v[i].to(device)
        return self

    def update_snapshot(self):
        self.theta_k = [param.clone().detach().to(self.device) for param in self.parameters]

    def receive_neighbor_snapshot(self, neighbor_id, neighbor_theta_k):
        self.neighbors_theta_k[neighbor_id] = [t.clone().detach().to(self.device) for t in neighbor_theta_k]

    def update_dual_variables(self, rho):
        for neighbor_theta in self.neighbors_theta_k.values():
            for p_tensor, my_theta, their_theta in zip(self.p, self.theta_k, neighbor_theta):
                p_tensor.add_(rho * (my_theta - their_theta))

class DiNNOCallback(BaseCallback):
    """
    Stable Baselines 3 Callback to integrate DiNNO/CADMM consensus.
    """
    def __init__(self, node_id, rho_init=0.0001, rho_final=0.01, rho_schedule_steps=1000000, registry=None, communication_freq=1000, target_params="actor", verbose=0):
        super().__init__(verbose)
        self.node_id = node_id
        # Start rho very small to prevent overpowering the initial RL gradients
        self.rho_init = rho_init
        self.rho_final = rho_final
        self.rho_schedule_steps = rho_schedule_steps
        self.rho = rho_init
        self.registry = registry 
        self.communication_freq = communication_freq
        self.target_params = target_params
        self.dinno_manager = None
        self.hooks = []

    def _init_callback(self):
        if self.target_params == "actor":
            if hasattr(self.model.policy, "actor"): # SAC
                params = list(self.model.policy.actor.parameters())
            elif hasattr(self.model.policy, "mlp_extractor") and hasattr(self.model.policy, "action_net"): # PPO
                # Crucial: Only sync the pi features, not the vf (critic) features!
                params = list(self.model.policy.mlp_extractor.policy_net.parameters()) + list(self.model.policy.action_net.parameters())
            else:
                raise ValueError("Could not find actor parameters.")
        else:
            params = list(self.model.policy.parameters())

        self.dinno_manager = DiNNOManager(params, node_id=self.node_id)
        self.dinno_manager.set_rho(self.rho)
        self.dinno_manager.to(self.model.device)

        for i, param in enumerate(params):
            if param.requires_grad:
                hook = self._make_hook(i)
                self.hooks.append(param.register_hook(hook))

    def _make_hook(self, idx):
        def hook(grad):
            c_grad = self.dinno_manager.get_consensus_grad(idx)
            
            # PROTECT ADAM: Clip the consensus gradient so it doesn't wipe out the RL task gradient.
            # We constrain the CADMM gradient norm to be at most 0.5 (SB3's default max_grad_norm)
            max_norm = 0.5
            c_norm = torch.norm(c_grad)
            if c_norm > max_norm:
                c_grad = c_grad * (max_norm / (c_norm + 1e-8))
                
            return grad + c_grad
        return hook

    def _on_rollout_end(self) -> bool:
        # PPO exchanges weights at the end of the rollout (before the optimization epochs begin)
        if isinstance(self.model, PPO):
            self._sync()
        return True

    def _on_step(self) -> bool:
        if self.rho_schedule_steps > 0:
            alpha = min(1.0, self.model.num_timesteps / self.rho_schedule_steps)
            self.rho = self.rho_init + alpha * (self.rho_final - self.rho_init)
            if self.dinno_manager:
                self.dinno_manager.set_rho(self.rho)

        # SAC updates every step, so we sync based on communication freq
        if not isinstance(self.model, PPO) and self.model.num_timesteps % self.communication_freq == 0:
            self._sync()
        return True

    def _sync(self):
        if self.dinno_manager is None:
            return
            
        self.dinno_manager.update_snapshot()
        
        if self.registry is not None:
            self.registry[self.node_id] = [t.cpu() for t in self.dinno_manager.theta_k]
            
            for other_id, snapshot in self.registry.items():
                if other_id != self.node_id:
                    self.dinno_manager.receive_neighbor_snapshot(other_id, snapshot)
        
        self.dinno_manager.update_dual_variables(self.rho)
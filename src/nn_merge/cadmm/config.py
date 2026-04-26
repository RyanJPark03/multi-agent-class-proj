from dataclasses import dataclass

@dataclass
class DiNNOConfig:
    rho_init: float = 0.001
    rho_final: float = 0.1
    rho_schedule_steps: int = 1000000
    communication_freq: int = 1000
    num_nodes: int = 2

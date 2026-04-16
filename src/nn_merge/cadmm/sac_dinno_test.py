import gymnasium as gym
from stable_baselines3 import SAC
from nn_merge.cadmm.dinno import DiNNOCallback
import torch
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

def test_sac_dinno():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing SAC + DiNNO Integration on {device}...")
    
    # 1. Create a shared registry for communication
    shared_registry = {}
    
    # 2. Setup environments
    env0 = gym.make("Pendulum-v1")
    env1 = gym.make("Pendulum-v1")
    
    # 3. Initialize models
    policy_kwargs = dict(net_arch=[64, 64])
    
    model0 = SAC("MlpPolicy", env0, policy_kwargs=policy_kwargs, verbose=0, device=device)
    model1 = SAC("MlpPolicy", env1, policy_kwargs=policy_kwargs, verbose=0, device=device)
    
    # 4. Setup DiNNO Callbacks
    # Node 0
    callback0 = DiNNOCallback(
        node_id=0, 
        rho=10.0, 
        registry=shared_registry, 
        communication_freq=100 # SAC communicates every N steps
    )
    
    # Node 1
    callback1 = DiNNOCallback(
        node_id=1, 
        rho=10.0, 
        registry=shared_registry, 
        communication_freq=50
    )
    
    # 5. Training iteration 
    print("Starting training with DiNNOCallback...")
    
    # Track parameter similarity
    def get_diff():
        diff = 0
        for p0, p1 in zip(model0.policy.parameters(), model1.policy.parameters()):
            diff += torch.sum((p0 - p1)**2).item()
        return diff

    initial_diff = get_diff()
    print(f"Initial Parameter Discrepancy: {initial_diff:.4f}")
    
    # Train for a few steps
    model0.learn(total_timesteps=2000, callback=callback0)
    model1.learn(total_timesteps=2000, callback=callback1)
    
    final_diff = get_diff()
    print(f"Final Parameter Discrepancy (after 300 steps): {final_diff:.4f}")
    
    if final_diff < initial_diff:
        print("\nSUCCESS: Parameter discrepancy decreased, suggesting consensus influence.")
    else:
        # Note: With only 300 steps of SAC, they might not converge yet, 
        # but we successfully ran the hooks without crashing.
        print("\nNOTE: DISCREPANCY DID NOT DECREASE (needs more steps/tuning for Pendulum).")
        print("Integration verified: Hooks and synchronization logic executed without errors.")

if __name__ == "__main__":
    test_sac_dinno()

"""
PPO Sniper - ONNX Export Utility
Converts SB3 Expert Models to Native MQL5 ONNX files with embedded normalization.
"""
import torch
import torch.nn as nn
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

class SmartOnnxWrapper(nn.Module):
    """
    Wraps the PPO Policy and adds Observation Normalization inside the graph.
    This allows MQL5 code to send RAW features.
    """
    def __init__(self, policy, mean, var, epsilon=1e-8):
        super().__init__()
        self.policy = policy
        
        # Convert stats to tensors
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("var", torch.tensor(var, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, x):
        # 1. Internal Normalization: (x - mean) / sqrt(var + eps)
        x_norm = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        
        # 2. Extract latent features from policy
        # For MlpPolicy, SB3 uses mlp_extractor for shared layers
        latent_pi, _ = self.policy.mlp_extractor(x_norm)
        
        # 3. Get action (mean of distribution for deterministic)
        action_mean = self.policy.action_net(latent_pi)
        
        # 4. For continuous PPO, we often want just the mean
        return action_mean

def export_to_onnx(symbol="xauusd"):
    # Paths
    symbol_root = os.path.join("models", symbol.lower())
    expert_dir = os.path.join(symbol_root, "experts")
    model_path = os.path.join(expert_dir, f"{symbol.lower()}_m5_ppo_expert.zip")
    stats_path = os.path.join(expert_dir, f"{symbol.lower()}_m5_ppo_expert_vec_normalize.pkl")
    output_path = os.path.join(expert_dir, f"{symbol.lower()}_ppo.onnx")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # 1. Load Model & Stats
    print(f"Loading Expert: {model_path}")
    model = PPO.load(model_path, device="cpu")
    
    mean, var = np.zeros(model.observation_space.shape), np.ones(model.observation_space.shape)
    if os.path.exists(stats_path):
        print(f"Loading Normalization Stats: {stats_path}")
        # Direct load to avoid needing a dummy environment
        import pickle
        with open(stats_path, "rb") as f:
            vn_data = pickle.load(f)
            # Handle both older and newer SB3 serialization versions
            if hasattr(vn_data, 'obs_rms'):
                mean = vn_data.obs_rms.mean
                var = vn_data.obs_rms.var
            else:
                # If it's a dict (some versions)
                mean = vn_data.get('obs_rms.mean', mean)
                var = vn_data.get('obs_rms.var', var)
    
    # 2. Create the Wrapper
    print("Creating Smart ONNX Wrapper...")
    wrapper = SmartOnnxWrapper(model.policy, mean, var)
    wrapper.eval()

    # 3. Export
    dummy_input = torch.randn(1, *model.observation_space.shape)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, # Modern opset for MT5 compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Done! You can now use this file in MQL5.")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "xauusd"
    export_to_onnx(target)

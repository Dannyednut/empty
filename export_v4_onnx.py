"""
Advanced ONNX Export with Multi-Action Rescaling
Version: V4 Sniper
"""
import torch
import torch.nn as nn
import pickle
import onnx
import os
from stable_baselines3 import PPO

class SniperOnnxWrapper(nn.Module):
    def __init__(self, policy, mean, var, epsilon=1e-8):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", torch.FloatTensor(mean))
        self.register_buffer("var", torch.FloatTensor(var))
        self.epsilon = epsilon

    def forward(self, x):
        # 1. Parity Normalization
        x_norm = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        
        # 2. Extract latent features
        latent_pi, _ = self.policy.mlp_extractor(x_norm)
        action_mean = self.policy.action_net(latent_pi)
        
        # 3. Multi-Head Rescaling
        # Head 0: Position [-1, 1]
        pos = torch.tanh(action_mean[:, 0:1])
        
        # Head 1: ATR Multiplier [0.5, 3.0]
        # Squashing to [-1, 1], then shifting to [0.5, 3.0]
        mult_raw = torch.tanh(action_mean[:, 1:2]) 
        # Shift formula: scaled = (raw + 1) / 2 * (max - min) + min
        mult_scaled = (mult_raw + 1.) / 2. * (3.0 - 0.5) + 0.5
        
        return torch.cat([pos, mult_scaled], dim=1)

def export_v4(symbol="xauusd"):
    model_path = f"models/{symbol}/sniper/{symbol}_sniper_v4_expert.zip"
    stats_path = f"models/{symbol}/sniper/{symbol}_sniper_v4_vec_normalize.pkl"
    output_path = f"models/{symbol}/sniper/{symbol}_sniper_v4.onnx"

    print(f"📦 Packaging Sniper V4 ONNX for {symbol}...")
    model = PPO.load(model_path)
    
    with open(stats_path, "rb") as f:
        vn = pickle.load(f)
        mean, var = vn.obs_rms.mean, vn.obs_rms.var

    wrapper = SniperOnnxWrapper(model.policy, mean, var)
    wrapper.eval()

    dummy_input = torch.randn(1, len(mean))
    torch.onnx.export(
        wrapper, dummy_input, output_path,
        export_params=True, opset_version=14,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # 4. Critical: Merge external weights for MetaTrader 5 (Single File Requirement)
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        print("⚠️ Split data detected. Merging weights into a single file...")
        import onnx
        # Load the model with its external data
        model_onnx = onnx.load(output_path, load_external_data=True)
        # Save it back as a single file
        onnx.save_model(model_onnx, output_path, save_as_external_data=False)
        # Cleanup
        os.remove(data_file)
        print("✅ SUCCESS: Model is now a single self-contained .onnx file.")
    
    print(f"🏁 Done! Sniper V4 Brain exported to {output_path}")

if __name__ == "__main__":
    export_v4()

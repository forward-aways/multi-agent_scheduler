"""
调试无人机模型输出
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_drone_env import MultiAgentDroneEnv

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.network(x)

# 创建环境
env_config = {
    'num_drones': 3,
    'max_speed': 2.0,
    'battery_capacity': 100.0,
    'payload_capacity': 5.0,
    'space_size': [100, 100, 50],
    'task_type': 'formation',
    'formation_type': 'triangle',
    'max_steps': 200
}

env = MultiAgentDroneEnv(env_config)
obs, _ = env.reset()

# 加载模型
model_dir = Path(project_root) / "models" / "multi_agent_drone" / "mappo" / "formation" / "triangle"
state_dim = env.observation_spaces['drone_0'].shape[0]
action_dim = env.action_spaces['drone_0'].shape[0]

print(f"状态维度：{state_dim}")
print(f"动作维度：{action_dim}")

for i in range(3):
    model = Actor(state_dim, action_dim, hidden_dim=256)
    model_path = model_dir / f'drone_{i}_agent.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    
    state_tensor = torch.FloatTensor(obs[f'drone_{i}']).unsqueeze(0)
    with torch.no_grad():
        output = model(state_tensor).squeeze(0).numpy()
    
    print(f"\n无人机{i}:")
    print(f"  模型输出 (Tanh 后): [{output[0]:.4f}, {output[1]:.4f}, {output[2]:.4f}]")
    print(f"  输出范围：[{output.min():.4f}, {output.max():.4f}]")
    print(f"  乘以 max_speed 后：[{output[0]*2:.4f}, {output[1]*2:.4f}, {output[2]*2:.4f}]")

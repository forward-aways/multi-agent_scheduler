"""
调试物流调度模型输出
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv

class Actor(nn.Module):
    """Actor 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# 创建环境
env_config = {
    'num_warehouses': 3,
    'num_vehicles': 5,
    'warehouse_capacity': 100,
    'vehicle_capacity': 20,
    'vehicle_speed': 5.0,
    'order_generation_rate': 2,
    'max_pending_orders': 15,
    'map_size': [100.0, 100.0],
    'max_steps': 200,
    'manual_mode': False
}

env = MultiAgentLogisticsEnv(env_config)
obs, _ = env.reset()

# 加载模型
model_dir = Path(project_root) / "models" / "multi_agent_logistics" / "mappo" / "best"

print("="*80)
print("物流调度模型输出分析")
print("="*80)

# 测试仓库智能体
print("\n【仓库智能体】")
for i in range(env.num_warehouses):
    agent_id = f'warehouse_{i}'
    model = Actor(env.observation_spaces[agent_id].shape[0], 
                  env.action_spaces[agent_id].n)
    model_path = model_dir / f'{agent_id}_agent.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    
    state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
    with torch.no_grad():
        logits = model(state_tensor).squeeze(0).numpy()
        probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
        action = np.argmax(logits)
    
    print(f"\n{agent_id}:")
    print(f"  观测维度：{obs[agent_id].shape}")
    print(f"  动作维度：{env.action_spaces[agent_id].n}")
    print(f"  动作 logits: [{logits[0]:.4f}, {logits[1]:.4f}, {logits[2]:.4f}]")
    print(f"  动作概率：[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")
    print(f"  选择动作：{action} (0=分配，1=拒绝，2=调整库存)")
    
    # 显示仓库状态
    print(f"  当前库存：{env.warehouse_inventory[i]:.1f}")
    print(f"  当前订单：{len(env.warehouse_orders[i])}")

# 测试车辆智能体
print("\n【车辆智能体】")
for i in range(env.num_vehicles):
    agent_id = f'vehicle_{i}'
    model = Actor(env.observation_spaces[agent_id].shape[0], 
                  env.action_spaces[agent_id].n)
    model_path = model_dir / f'{agent_id}_agent.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    
    state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
    with torch.no_grad():
        logits = model(state_tensor).squeeze(0).numpy()
        probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
        action = np.argmax(logits)
    
    print(f"\n{agent_id}:")
    print(f"  观测维度：{obs[agent_id].shape}")
    print(f"  动作维度：{env.action_spaces[agent_id].n}")
    print(f"  动作 logits: [{logits[0]:.4f}, {logits[1]:.4f}, {logits[2]:.4f}, {logits[3]:.4f}]")
    print(f"  动作概率：[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}, {probs[3]:.4f}]")
    print(f"  选择动作：{action} (0=空闲，1=去仓库，2=装货/配送，3=返回)")
    
    # 显示车辆状态
    status_map = {0: '空闲', 1: '去仓库', 2: '装货', 3: '去配送', 4: '卸货', 5: '返回'}
    print(f"  当前状态：{status_map[env.vehicle_status[i]]}")
    print(f"  当前载货：{env.vehicle_cargo[i]:.1f}/{env.vehicle_capacity}")
    print(f"  当前位置：[{env.vehicle_positions[i][0]:.1f}, {env.vehicle_positions[i][1]:.1f}]")

print("\n" + "="*80)
print("环境状态总览")
print("="*80)
print(f"待处理订单：{len(env.pending_orders)}")
print(f"配送中订单：{len(env.delivering_orders)}")
print(f"已完成订单：{env.completed_orders}")
print(f"失败订单：{env.failed_orders}")

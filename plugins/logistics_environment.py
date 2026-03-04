"""
物流调度环境插件
将现有的物流环境封装为插件
"""

import sys
sys.path.insert(0, 'd:\\graduation_project\\multi_agent_scheduler')

from typing import Dict, Any
import numpy as np

from core.plugin_interface import EnvironmentPlugin
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv


class LogisticsEnvironmentPlugin(EnvironmentPlugin):
    """物流环境插件"""
    
    def __init__(self):
        super().__init__(name="logistics_environment", version="1.0.0")
        self.env_type = "logistics_scheduling"
        self.env = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化环境"""
        try:
            self.config = config
            # 构造环境配置
            env_config = {
                'num_vehicles': config.get('num_vehicles', 5),
                'num_warehouses': config.get('num_warehouses', 3),
                'num_customers': config.get('num_customers', 10),
                'max_steps': config.get('max_steps', 200),
                'world_size': config.get('world_size', 100),
                'vehicle_capacity': config.get('vehicle_capacity', 100.0),
                'order_generation_rate': config.get('order_generation_rate', 2)
            }
            self.env = MultiAgentLogisticsEnv(env_config)
            self.is_active = True
            return True
        except Exception as e:
            print(f"初始化物流环境失败: {e}")
            return False
    
    def create_environment(self, config: Dict[str, Any]):
        """创建环境实例"""
        env_config = {
            'num_vehicles': config.get('num_vehicles', 5),
            'num_warehouses': config.get('num_warehouses', 3),
            'num_customers': config.get('num_customers', 10),
            'max_steps': config.get('max_steps', 200),
            'world_size': config.get('world_size', 100),
            'vehicle_capacity': config.get('vehicle_capacity', 100.0),
            'order_generation_rate': config.get('order_generation_rate', 2)
        }
        return MultiAgentLogisticsEnv(env_config)
    
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观测空间"""
        if self.env is None:
            return {}
        
        return {
            'num_vehicles': self.env.num_vehicles,
            'num_warehouses': self.env.num_warehouses,
            'observation_dim': self.env.observation_dim
        }
    
    def get_action_space(self) -> Dict[str, Any]:
        """获取动作空间"""
        if self.env is None:
            return {}
        
        return {
            'num_actions': self.env.num_actions,
            'action_space_type': 'discrete'
        }
    
    def reset(self) -> tuple:
        """重置环境"""
        if self.env is None:
            return None, {}
        return self.env.reset()
    
    def step(self, actions: Dict[str, int]) -> tuple:
        """执行一步"""
        if self.env is None:
            return None, {}, True, False, {}
        
        # 转换动作格式
        action_dict = {}
        for i in range(self.env.num_vehicles):
            agent_id = f'vehicle_{i}'
            action_dict[agent_id] = actions.get(agent_id, 0)
        
        return self.env.step(action_dict)
    
    def shutdown(self) -> bool:
        """关闭环境"""
        self.env = None
        self.is_active = False
        return True

"""
无人机环境插件
将现有的无人机环境封装为插件
"""

import sys
sys.path.insert(0, 'd:\\graduation_project\\multi_agent_scheduler')

from typing import Dict, Any
import numpy as np

from core.plugin_interface import EnvironmentPlugin
from environments.multi_agent_drone_env import MultiAgentDroneEnv


class DroneEnvironmentPlugin(EnvironmentPlugin):
    """无人机环境插件"""
    
    def __init__(self):
        super().__init__(name="drone_environment", version="1.0.0")
        self.env_type = "drone_formation"
        self.env = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化环境"""
        try:
            self.config = config
            # 构造环境配置
            env_config = {
                'num_drones': config.get('num_drones', 3),
                'formation_type': config.get('formation_type', 'triangle'),
                'max_steps': config.get('max_steps', 500),
                'world_size': config.get('world_size', 1000),
                'dt': config.get('dt', 0.1),
                'max_speed': config.get('max_speed', 10.0)
            }
            self.env = MultiAgentDroneEnv(env_config)
            self.is_active = True
            return True
        except Exception as e:
            print(f"初始化无人机环境失败: {e}")
            return False
    
    def create_environment(self, config: Dict[str, Any]):
        """创建环境实例"""
        env_config = {
            'num_drones': config.get('num_drones', 3),
            'formation_type': config.get('formation_type', 'triangle'),
            'max_steps': config.get('max_steps', 500),
            'world_size': config.get('world_size', 1000),
            'dt': config.get('dt', 0.1),
            'max_speed': config.get('max_speed', 10.0)
        }
        return MultiAgentDroneEnv(env_config)
    
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观测空间"""
        if self.env is None:
            return {}
        
        return {
            'num_drones': self.env.num_drones,
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
        for i in range(self.env.num_drones):
            agent_id = f'drone_{i}'
            action_dict[agent_id] = actions.get(agent_id, 13)  # 默认悬停
        
        return self.env.step(action_dict)
    
    def shutdown(self) -> bool:
        """关闭环境"""
        self.env = None
        self.is_active = False
        return True

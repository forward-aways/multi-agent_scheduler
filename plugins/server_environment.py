"""
服务器调度环境插件
将现有的服务器环境封装为插件
"""

import sys
sys.path.insert(0, 'd:\\graduation_project\\multi_agent_scheduler')

from typing import Dict, Any
import numpy as np

from core.plugin_interface import EnvironmentPlugin
from environments.multi_agent_server_env import MultiAgentServerEnv


class ServerEnvironmentPlugin(EnvironmentPlugin):
    """服务器环境插件"""
    
    def __init__(self):
        super().__init__(name="server_environment", version="1.0.0")
        self.env_type = "server_scheduling"
        self.env = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化环境"""
        try:
            self.config = config
            # 构造环境配置
            env_config = {
                'num_servers': config.get('num_servers', 5),
                'server_cpu_capacity': config.get('server_cpu_capacity', 100.0),
                'server_memory_capacity': config.get('server_memory_capacity', 100.0),
                'server_max_tasks': config.get('server_max_tasks', 10),
                'task_generation_rate': config.get('task_generation_rate', 3),
                'max_pending_tasks': config.get('max_pending_tasks', 20),
                'max_steps': config.get('max_steps', 100)
            }
            self.env = MultiAgentServerEnv(env_config)
            self.is_active = True
            return True
        except Exception as e:
            print(f"初始化服务器环境失败: {e}")
            return False
    
    def create_environment(self, config: Dict[str, Any]):
        """创建环境实例"""
        env_config = {
            'num_servers': config.get('num_servers', 5),
            'server_cpu_capacity': config.get('server_cpu_capacity', 100.0),
            'server_memory_capacity': config.get('server_memory_capacity', 100.0),
            'server_max_tasks': config.get('server_max_tasks', 10),
            'task_generation_rate': config.get('task_generation_rate', 3),
            'max_pending_tasks': config.get('max_pending_tasks', 20),
            'max_steps': config.get('max_steps', 100)
        }
        return MultiAgentServerEnv(env_config)
    
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观测空间"""
        if self.env is None:
            return {}
        
        return {
            'shape': self.env.observation_space.shape,
            'dtype': str(self.env.observation_space.dtype),
            'low': self.env.observation_space.low.tolist(),
            'high': self.env.observation_space.high.tolist()
        }
    
    def get_action_space(self) -> Dict[str, Any]:
        """获取动作空间"""
        if self.env is None:
            return {}
        
        return {
            'n': self.env.action_space.n,
            'dtype': str(self.env.action_space.dtype)
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
        action_array = np.array([actions.get(f'server_{i}', 0) 
                                for i in range(self.env.num_servers)])
        
        return self.env.step(action_array)
    
    def shutdown(self) -> bool:
        """关闭环境"""
        self.env = None
        self.is_active = False
        return True

"""
随机策略插件
作为基准策略用于对比
"""

from typing import Dict, Any
import numpy as np

from core.plugin_interface import StrategyPlugin


class RandomStrategyPlugin(StrategyPlugin):
    """随机策略插件"""
    
    def __init__(self):
        super().__init__(name="random_strategy", version="1.0.0")
        self.strategy_type = "random"
        self.num_actions = 27
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化策略"""
        self.config = config
        self.num_actions = config.get('num_actions', 27)
        self.is_active = True
        return True
    
    def make_decision(self, observation: np.ndarray, agent_id: str) -> int:
        """随机决策"""
        return np.random.randint(0, self.num_actions)
    
    def train(self, env, episodes: int = 100) -> Dict[str, Any]:
        """随机策略不需要训练"""
        return {
            'success': True,
            'message': '随机策略无需训练',
            'episodes': 0
        }
    
    def load_model(self, model_path: str) -> bool:
        """随机策略不需要模型"""
        return True
    
    def save_model(self, model_path: str) -> bool:
        """随机策略不需要保存模型"""
        return True
    
    def shutdown(self) -> bool:
        """关闭策略"""
        self.is_active = False
        return True

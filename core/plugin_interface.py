"""
插件接口定义模块
定义所有插件必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class PluginInterface(ABC):
    """插件基础接口"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.config = {}
        self.is_active = False
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化插件
        
        Args:
            config: 配置参数
            
        Returns:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行插件功能
        
        Args:
            context: 执行上下文
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        关闭插件
        
        Returns:
            关闭是否成功
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            'name': self.name,
            'version': self.version,
            'is_active': self.is_active,
            'config': self.config
        }


class StrategyPlugin(PluginInterface):
    """策略插件接口 - 用于实现不同的调度策略"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.strategy_type = "unknown"
    
    @abstractmethod
    def make_decision(self, observation: np.ndarray, agent_id: str) -> int:
        """
        做出调度决策
        
        Args:
            observation: 环境观测
            agent_id: 智能体ID
            
        Returns:
            动作决策
        """
        pass
    
    @abstractmethod
    def train(self, env, episodes: int = 100) -> Dict[str, Any]:
        """
        训练策略
        
        Args:
            env: 训练环境
            episodes: 训练回合数
            
        Returns:
            训练结果统计
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载是否成功
        """
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> bool:
        """
        保存模型
        
        Args:
            model_path: 保存路径
            
        Returns:
            保存是否成功
        """
        pass
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行策略决策"""
        observation = context.get('observation')
        agent_id = context.get('agent_id')
        
        if observation is None or agent_id is None:
            return {'success': False, 'error': 'Missing observation or agent_id'}
        
        action = self.make_decision(observation, agent_id)
        return {'success': True, 'action': action}


class EnvironmentPlugin(PluginInterface):
    """环境插件接口 - 用于支持不同的应用场景"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.env_type = "unknown"
        self.env = None
    
    @abstractmethod
    def create_environment(self, config: Dict[str, Any]) -> Any:
        """
        创建环境实例
        
        Args:
            config: 环境配置
            
        Returns:
            环境实例
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """
        获取观测空间定义
        
        Returns:
            观测空间信息
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """
        获取动作空间定义
        
        Returns:
            动作空间信息
        """
        pass
    
    @abstractmethod
    def reset(self) -> tuple:
        """
        重置环境
        
        Returns:
            (observation, info)
        """
        pass
    
    @abstractmethod
    def step(self, actions: Dict[str, int]) -> tuple:
        """
        执行一步
        
        Args:
            actions: 智能体动作
            
        Returns:
            (observation, reward, done, truncated, info)
        """
        pass
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行环境操作"""
        operation = context.get('operation')
        
        if operation == 'reset':
            obs, info = self.reset()
            return {'success': True, 'observation': obs, 'info': info}
        elif operation == 'step':
            actions = context.get('actions', {})
            obs, reward, done, truncated, info = self.step(actions)
            return {
                'success': True,
                'observation': obs,
                'reward': reward,
                'done': done,
                'truncated': truncated,
                'info': info
            }
        else:
            return {'success': False, 'error': f'Unknown operation: {operation}'}


class EvaluationPlugin(PluginInterface):
    """评估插件接口 - 用于性能评估"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.metrics = []
    
    @abstractmethod
    def evaluate(self, env, strategy, episodes: int = 100) -> Dict[str, Any]:
        """
        评估策略性能
        
        Args:
            env: 评估环境
            strategy: 评估策略
            episodes: 评估回合数
            
        Returns:
            评估指标
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[str]:
        """
        获取支持的评估指标
        
        Returns:
            指标名称列表
        """
        pass
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行评估"""
        env = context.get('env')
        strategy = context.get('strategy')
        episodes = context.get('episodes', 100)
        
        if env is None or strategy is None:
            return {'success': False, 'error': 'Missing env or strategy'}
        
        results = self.evaluate(env, strategy, episodes)
        return {'success': True, 'results': results}

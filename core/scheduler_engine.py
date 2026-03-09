"""
调度引擎核心模块
作为系统中枢，统一协调各功能单元
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import logging
from pathlib import Path

from .plugin_manager import PluginManager
from .plugin_interface import StrategyPlugin, EnvironmentPlugin, EvaluationPlugin

logger = logging.getLogger(__name__)


class SchedulerEngine:
    """
    调度引擎核心类
    
    功能：
    1. 插件管理：加载、初始化、切换插件
    2. 调度协调：统一管理各功能单元的调度流程
    3. 运行时切换：支持策略和环境的动态切换
    4. 批量评估：支持多场景批量评估模式
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化调度引擎
        
        Args:
            config: 引擎配置
        """
        self.config = config or {}
        self.plugin_manager = PluginManager(
            plugin_dirs=self.config.get('plugin_dirs', ['plugins'])
        )
        
        # 当前状态
        self.current_environment: Optional[EnvironmentPlugin] = None
        self.current_strategy: Optional[StrategyPlugin] = None
        self.current_evaluation: Optional[EvaluationPlugin] = None
        
        # 运行状态
        self.is_running = False
        self.episode_count = 0
        self.step_count = 0
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'on_episode_start': [],
            'on_episode_end': [],
            'on_step': [],
            'on_strategy_switch': [],
            'on_environment_switch': []
        }
        
        logger.info("调度引擎初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化引擎
        
        Returns:
            初始化是否成功
        """
        try:
            # 发现插件
            discovered = self.plugin_manager.discover_plugins()
            logger.info(f"发现 {len(discovered)} 个插件")
            
            # 加载所有发现的插件
            for plugin_name in discovered:
                self.plugin_manager.load_plugin(plugin_name)
            
            logger.info("调度引擎初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"调度引擎初始化失败: {e}")
            return False
    
    def load_scenario(self, scenario_config: Dict[str, Any]) -> bool:
        """
        加载场景配置
        
        Args:
            scenario_config: 场景配置
                {
                    'environment': '环境插件名称',
                    'strategy': '策略插件名称',
                    'env_config': {},  # 环境配置
                    'strategy_config': {}  # 策略配置
                }
        
        Returns:
            加载是否成功
        """
        try:
            # 加载环境
            env_name = scenario_config.get('environment')
            if env_name:
                if not self.plugin_manager.initialize_plugin(
                    env_name, 
                    scenario_config.get('env_config', {})
                ):
                    logger.error(f"初始化环境插件 {env_name} 失败")
                    return False
                
                self.plugin_manager.activate_plugin(env_name, 'environment')
                self.current_environment = self.plugin_manager.get_active_plugin('environment')
                
                if self.current_environment is None:
                    logger.error(f"激活环境插件 {env_name} 失败")
                    return False
                
                logger.info(f"已加载环境: {env_name}")
            
            # 加载策略
            strategy_name = scenario_config.get('strategy')
            if strategy_name:
                if not self.plugin_manager.initialize_plugin(
                    strategy_name,
                    scenario_config.get('strategy_config', {})
                ):
                    logger.error(f"初始化策略插件 {strategy_name} 失败")
                    return False
                
                self.plugin_manager.activate_plugin(strategy_name, 'strategy')
                self.current_strategy = self.plugin_manager.get_active_plugin('strategy')
                
                if self.current_strategy is None:
                    logger.error(f"激活策略插件 {strategy_name} 失败")
                    return False
                
                logger.info(f"已加载策略: {strategy_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载场景失败: {e}")
            return False
    
    def switch_strategy(self, strategy_name: str, config: Dict[str, Any] = None) -> bool:
        """
        运行时切换策略
        
        Args:
            strategy_name: 新策略名称
            config: 策略配置
            
        Returns:
            切换是否成功
        """
        try:
            # 初始化新策略
            if not self.plugin_manager.initialize_plugin(strategy_name, config or {}):
                logger.error(f"初始化策略 {strategy_name} 失败")
                return False
            
            # 切换策略
            if self.plugin_manager.switch_strategy(strategy_name):
                self.current_strategy = self.plugin_manager.get_active_plugin('strategy')
                
                # 触发回调
                self._trigger_callbacks('on_strategy_switch', {
                    'new_strategy': strategy_name
                })
                
                logger.info(f"成功切换到策略: {strategy_name}")
                return True
            else:
                logger.error(f"切换策略 {strategy_name} 失败")
                return False
                
        except Exception as e:
            logger.error(f"切换策略时出错: {e}")
            return False
    
    def switch_environment(self, env_name: str, config: Dict[str, Any] = None) -> bool:
        """
        运行时切换环境
        
        Args:
            env_name: 新环境名称
            config: 环境配置
            
        Returns:
            切换是否成功
        """
        try:
            # 初始化新环境
            if not self.plugin_manager.initialize_plugin(env_name, config or {}):
                logger.error(f"初始化环境 {env_name} 失败")
                return False
            
            # 切换环境
            if self.plugin_manager.switch_environment(env_name):
                self.current_environment = self.plugin_manager.get_active_plugin('environment')
                
                # 触发回调
                self._trigger_callbacks('on_environment_switch', {
                    'new_environment': env_name
                })
                
                logger.info(f"成功切换到环境: {env_name}")
                return True
            else:
                logger.error(f"切换环境 {env_name} 失败")
                return False
                
        except Exception as e:
            logger.error(f"切换环境时出错: {e}")
            return False
    
    def run_episode(self, max_steps: int = None, render: bool = False) -> Dict[str, Any]:
        """
        运行一个回合
        
        Args:
            max_steps: 最大步数
            render: 是否渲染
            
        Returns:
            回合结果
        """
        if self.current_environment is None or self.current_strategy is None:
            return {'success': False, 'error': 'Environment or strategy not set'}
        
        try:
            self.episode_count += 1
            episode_data = {
                'episode': self.episode_count,
                'steps': 0,
                'total_reward': 0,
                'rewards': [],
                'delays': [],  # 延迟数据
                'agent_utilization': {},  # 智能体利用率
                'tasks': {'total': 0, 'completed': 0, 'failed': 0}  # 任务统计
            }
            
            # 触发回合开始回调
            self._trigger_callbacks('on_episode_start', episode_data)
            
            # 重置环境
            reset_result = self.plugin_manager.execute_plugin(
                self.current_environment.name,
                {'operation': 'reset'}
            )
            
            if not reset_result.get('success'):
                return {'success': False, 'error': 'Failed to reset environment'}
            
            observation = reset_result['observation']
            info = reset_result['info']
            
            # 运行回合
            done = False
            truncated = False
            step = 0
            max_steps = max_steps or 1000
            
            while not done and not truncated and step < max_steps:
                # 获取动作
                actions = {}
                
                # 处理不同格式的观测（字典或numpy数组）
                if isinstance(observation, dict):
                    # 多智能体环境，观测是字典
                    agent_ids = list(observation.keys())
                else:
                    # 单智能体环境或数组格式，创建默认agent_id
                    agent_ids = ['agent_0']
                    observation = {'agent_0': observation}
                
                for agent_id in agent_ids:
                    action_result = self.plugin_manager.execute_plugin(
                        self.current_strategy.name,
                        {
                            'observation': observation[agent_id],
                            'agent_id': agent_id
                        }
                    )
                    
                    if action_result.get('success'):
                        actions[agent_id] = action_result['action']
                
                # 执行动作
                step_result = self.plugin_manager.execute_plugin(
                    self.current_environment.name,
                    {
                        'operation': 'step',
                        'actions': actions
                    }
                )
                
                if not step_result.get('success'):
                    break
                
                observation = step_result['observation']
                reward = step_result['reward']
                done_dict = step_result['done']
                truncated_dict = step_result['truncated']
                info = step_result['info']
                
                # 处理多智能体环境的done和truncated（字典格式）
                # 如果所有智能体都完成，则认为回合结束
                if isinstance(done_dict, dict):
                    done = all(done_dict.values())
                else:
                    done = done_dict
                    
                if isinstance(truncated_dict, dict):
                    truncated = all(truncated_dict.values())
                else:
                    truncated = truncated_dict
                
                # 记录数据
                episode_data['steps'] += 1
                if isinstance(reward, dict):
                    episode_data['total_reward'] += sum(reward.values())
                else:
                    episode_data['total_reward'] += reward
                episode_data['rewards'].append(reward)
                
                # 从info中提取额外数据（如果环境提供）
                # info 是多智能体字典格式: {'server_0': {...}, 'server_1': {...}}
                if isinstance(info, dict):
                    # 遍历每个智能体的info
                    for agent_id, agent_info in info.items():
                        if isinstance(agent_info, dict):
                            # 提取延迟数据
                            if 'delays' in agent_info:
                                episode_data['delays'].extend(agent_info['delays'])
                            # 提取智能体利用率
                            if 'agent_utilization' in agent_info:
                                episode_data['agent_utilization'].update(agent_info['agent_utilization'])
                            # 提取任务统计
                            if 'tasks' in agent_info:
                                task_info = agent_info['tasks']
                                episode_data['tasks']['total'] += task_info.get('total', 0)
                                episode_data['tasks']['completed'] += task_info.get('completed', 0)
                                episode_data['tasks']['failed'] += task_info.get('failed', 0)
                
                # 触发步进回调
                self._trigger_callbacks('on_step', {
                    'episode': self.episode_count,
                    'step': step,
                    'observation': observation,
                    'actions': actions,
                    'reward': reward,
                    'info': info
                })
                
                step += 1
                self.step_count += 1
            
            # 触发回合结束回调
            self._trigger_callbacks('on_episode_end', episode_data)
            
            return {
                'success': True,
                'episode_data': episode_data
            }
            
        except Exception as e:
            logger.error(f"运行回合时出错: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_batch_evaluation(
        self,
        scenarios: List[Dict[str, Any]],
        episodes_per_scenario: int = 10,
        callbacks: Dict[str, Callable] = None
    ) -> Dict[str, Any]:
        """
        批量评估模式
        
        Args:
            scenarios: 场景配置列表
            episodes_per_scenario: 每个场景的评估回合数
            callbacks: 回调函数字典
            
        Returns:
            评估结果
        """
        logger.info(f"开始批量评估: {len(scenarios)} 个场景，每个 {episodes_per_scenario} 回合")
        
        # 注册回调
        if callbacks:
            for event, callback in callbacks.items():
                self.register_callback(event, callback)
        
        evaluation_results = {
            'total_scenarios': len(scenarios),
            'episodes_per_scenario': episodes_per_scenario,
            'scenarios': []
        }
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"评估场景 {i+1}/{len(scenarios)}: {scenario.get('name', f'Scenario_{i}')}")
            
            # 加载场景
            if not self.load_scenario(scenario):
                logger.error(f"加载场景 {i} 失败，跳过")
                continue
            
            scenario_results = {
                'scenario_id': i,
                'scenario_name': scenario.get('name', f'Scenario_{i}'),
                'episodes': []
            }
            
            # 运行多个回合
            for episode in range(episodes_per_scenario):
                result = self.run_episode()
                
                if result.get('success'):
                    scenario_results['episodes'].append(result['episode_data'])
                else:
                    logger.warning(f"回合 {episode} 运行失败: {result.get('error')}")
            
            # 计算统计信息
            if scenario_results['episodes']:
                total_rewards = [ep['total_reward'] for ep in scenario_results['episodes']]
                steps = [ep['steps'] for ep in scenario_results['episodes']]
                
                scenario_results['statistics'] = {
                    'avg_reward': np.mean(total_rewards),
                    'std_reward': np.std(total_rewards),
                    'avg_steps': np.mean(steps),
                    'success_rate': len(scenario_results['episodes']) / episodes_per_scenario
                }
            
            evaluation_results['scenarios'].append(scenario_results)
        
        logger.info("批量评估完成")
        return evaluation_results
    
    def register_callback(self, event: str, callback: Callable):
        """
        注册回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.debug(f"注册回调: {event}")
    
    def unregister_callback(self, event: str, callback: Callable):
        """
        注销回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            logger.debug(f"注销回调: {event}")
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]):
        """
        触发回调
        
        Args:
            event: 事件名称
            data: 事件数据
        """
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"执行回调 {event} 时出错: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取引擎状态
        
        Returns:
            状态信息
        """
        return {
            'is_running': self.is_running,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'current_environment': self.current_environment.name if self.current_environment else None,
            'current_strategy': self.current_strategy.name if self.current_strategy else None,
            'loaded_plugins': len(self.plugin_manager.plugins),
            'active_plugins': len(self.plugin_manager.active_plugins)
        }
    
    def shutdown(self):
        """关闭引擎"""
        logger.info("关闭调度引擎...")
        
        self.is_running = False
        self.plugin_manager.shutdown_all()
        
        self.current_environment = None
        self.current_strategy = None
        self.current_evaluation = None
        
        logger.info("调度引擎已关闭")

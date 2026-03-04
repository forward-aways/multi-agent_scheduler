"""
调度器API模块
提供标准化的外部调用接口
支持HTTP REST API和Python API两种方式
"""

from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json
import logging
from datetime import datetime
import threading
import queue

# 尝试导入Flask，如果没有安装则使用模拟模式
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask未安装，API服务器将以模拟模式运行")

from core.scheduler_engine import SchedulerEngine
from core.plugin_manager import PluginManager

logger = logging.getLogger(__name__)


class SchedulerAPI:
    """
    调度器API类
    
    提供标准化的任务分配服务接口
    """
    
    def __init__(self, scheduler_engine: SchedulerEngine = None):
        """
        初始化API
        
        Args:
            scheduler_engine: 调度引擎实例（可选）
        """
        self.engine = scheduler_engine or SchedulerEngine()
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False
        
        # 请求统计
        self.request_count = 0
        self.error_count = 0
        
        logger.info("调度器API初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化API和引擎
        
        Returns:
            初始化是否成功
        """
        try:
            if not self.engine.initialize():
                logger.error("引擎初始化失败")
                return False
            
            self.is_running = True
            logger.info("调度器API初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化API时出错: {e}")
            return False
    
    # ==================== 场景管理API ====================
    
    def load_scenario(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载场景
        
        API端点: POST /api/scenario/load
        
        Args:
            scenario_config: 场景配置
                {
                    'name': '场景名称',
                    'environment': '环境插件名',
                    'strategy': '策略插件名',
                    'env_config': {},
                    'strategy_config': {}
                }
        
        Returns:
            加载结果
        """
        try:
            success = self.engine.load_scenario(scenario_config)
            
            return {
                'success': success,
                'message': '场景加载成功' if success else '场景加载失败',
                'scenario': scenario_config.get('name'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"加载场景时出错: {e}")
            return {
                'success': False,
                'message': f'加载场景失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_available_scenarios(self) -> Dict[str, Any]:
        """
        获取可用场景列表
        
        API端点: GET /api/scenarios
        
        Returns:
            场景列表
        """
        try:
            # 获取所有插件
            plugins = self.engine.plugin_manager.list_plugins()
            
            environments = [p for p in plugins if p.get('category') == 'environment']
            strategies = [p for p in plugins if p.get('category') == 'strategy']
            
            return {
                'success': True,
                'environments': environments,
                'strategies': strategies,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取场景列表时出错: {e}")
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ==================== 策略切换API ====================
    
    def switch_strategy(self, strategy_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        切换策略
        
        API端点: POST /api/strategy/switch
        
        Args:
            strategy_name: 策略名称
            config: 策略配置
        
        Returns:
            切换结果
        """
        try:
            success = self.engine.switch_strategy(strategy_name, config)
            
            return {
                'success': success,
                'message': f'策略切换{"成功" if success else "失败"}',
                'strategy': strategy_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"切换策略时出错: {e}")
            return {
                'success': False,
                'message': f'切换策略失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_current_strategy(self) -> Dict[str, Any]:
        """
        获取当前策略信息
        
        API端点: GET /api/strategy/current
        
        Returns:
            策略信息
        """
        try:
            strategy = self.engine.current_strategy
            
            if strategy is None:
                return {
                    'success': True,
                    'strategy': None,
                    'message': '当前未设置策略',
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'success': True,
                'strategy': strategy.get_info(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取当前策略时出错: {e}")
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ==================== 任务分配API ====================
    
    def request_task_allocation(
        self,
        task_info: Dict[str, Any],
        agent_id: str = None
    ) -> Dict[str, Any]:
        """
        请求任务分配服务
        
        API端点: POST /api/task/allocate
        
        Args:
            task_info: 任务信息
                {
                    'task_id': '任务ID',
                    'task_type': '任务类型',
                    'requirements': {},  # 任务需求
                    'priority': 1,  # 优先级
                    'deadline': '截止时间'
                }
            agent_id: 请求分配的智能体ID（可选）
        
        Returns:
            分配结果
        """
        self.request_count += 1
        
        try:
            if self.engine.current_strategy is None:
                return {
                    'success': False,
                    'message': '未加载策略，无法分配任务',
                    'task_id': task_info.get('task_id'),
                    'timestamp': datetime.now().isoformat()
                }
            
            # 构造观测（简化版本，实际应根据环境构造）
            observation = self._construct_observation_from_task(task_info)
            
            # 获取动作决策
            result = self.engine.plugin_manager.execute_plugin(
                self.engine.current_strategy.name,
                {
                    'observation': observation,
                    'agent_id': agent_id or 'agent_0'
                }
            )
            
            if result.get('success'):
                return {
                    'success': True,
                    'task_id': task_info.get('task_id'),
                    'allocation': {
                        'action': result['action'],
                        'agent_id': agent_id,
                        'strategy': self.engine.current_strategy.name
                    },
                    'timestamp': datetime.now().isoformat()
                }
            else:
                self.error_count += 1
                return {
                    'success': False,
                    'message': result.get('error', '分配失败'),
                    'task_id': task_info.get('task_id'),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"任务分配时出错: {e}")
            return {
                'success': False,
                'message': f'任务分配失败: {str(e)}',
                'task_id': task_info.get('task_id'),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_allocate_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        批量任务分配
        
        API端点: POST /api/task/batch_allocate
        
        Args:
            tasks: 任务列表
        
        Returns:
            批量分配结果
        """
        results = []
        
        for task in tasks:
            result = self.request_task_allocation(task)
            results.append(result)
        
        success_count = sum(1 for r in results if r.get('success'))
        
        return {
            'success': True,
            'total_tasks': len(tasks),
            'successful_allocations': success_count,
            'failed_allocations': len(tasks) - success_count,
            'allocations': results,
            'timestamp': datetime.now().isoformat()
        }
    
    # ==================== 状态查询API ====================
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        获取引擎状态
        
        API端点: GET /api/status
        
        Returns:
            状态信息
        """
        try:
            status = self.engine.get_status()
            status['api'] = {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(1, self.request_count)
            }
            status['timestamp'] = datetime.now().isoformat()
            
            return {
                'success': True,
                'status': status
            }
            
        except Exception as e:
            logger.error(f"获取状态时出错: {e}")
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_plugin_list(self, category: str = None) -> Dict[str, Any]:
        """
        获取插件列表
        
        API端点: GET /api/plugins
        
        Args:
            category: 插件类别过滤器
        
        Returns:
            插件列表
        """
        try:
            plugins = self.engine.plugin_manager.list_plugins(category)
            
            return {
                'success': True,
                'plugins': plugins,
                'count': len(plugins),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取插件列表时出错: {e}")
            return {
                'success': False,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ==================== 辅助方法 ====================
    
    def _construct_observation_from_task(self, task_info: Dict[str, Any]) -> Any:
        """
        从任务信息构造观测
        
        Args:
            task_info: 任务信息
        
        Returns:
            观测数据
        """
        # 简化实现，实际应根据环境类型构造合适的观测
        import numpy as np
        
        # 构造一个基本的观测向量
        priority = task_info.get('priority', 1)
        task_type = hash(task_info.get('task_type', 'default')) % 10
        
        observation = np.array([
            priority / 10.0,  # 归一化优先级
            task_type / 10.0,  # 任务类型编码
            0.5,  # 占位符
            0.5,  # 占位符
            0.5,  # 占位符
        ], dtype=np.float32)
        
        return observation
    
    def shutdown(self):
        """关闭API"""
        logger.info("关闭调度器API...")
        self.is_running = False
        self.engine.shutdown()
        logger.info("调度器API已关闭")


class APIServer:
    """
    API服务器类
    
    提供HTTP REST API服务
    """
    
    def __init__(self, scheduler_api: SchedulerAPI = None, host: str = '0.0.0.0', port: int = 5000):
        """
        初始化API服务器
        
        Args:
            scheduler_api: 调度器API实例
            host: 主机地址
            port: 端口号
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask未安装，无法启动API服务器。请运行: pip install flask")
        
        self.api = scheduler_api or SchedulerAPI()
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        
        # 注册路由
        self._register_routes()
        
        logger.info(f"API服务器初始化完成 ({host}:{port})")
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            return jsonify(self.api.get_engine_status())
        
        @self.app.route('/api/scenarios', methods=['GET'])
        def get_scenarios():
            return jsonify(self.api.get_available_scenarios())
        
        @self.app.route('/api/scenario/load', methods=['POST'])
        def load_scenario():
            config = request.get_json()
            return jsonify(self.api.load_scenario(config))
        
        @self.app.route('/api/strategy/current', methods=['GET'])
        def get_current_strategy():
            return jsonify(self.api.get_current_strategy())
        
        @self.app.route('/api/strategy/switch', methods=['POST'])
        def switch_strategy():
            data = request.get_json()
            strategy_name = data.get('strategy_name')
            config = data.get('config', {})
            return jsonify(self.api.switch_strategy(strategy_name, config))
        
        @self.app.route('/api/task/allocate', methods=['POST'])
        def allocate_task():
            data = request.get_json()
            task_info = data.get('task_info', {})
            agent_id = data.get('agent_id')
            return jsonify(self.api.request_task_allocation(task_info, agent_id))
        
        @self.app.route('/api/task/batch_allocate', methods=['POST'])
        def batch_allocate():
            data = request.get_json()
            tasks = data.get('tasks', [])
            return jsonify(self.api.batch_allocate_tasks(tasks))
        
        @self.app.route('/api/plugins', methods=['GET'])
        def get_plugins():
            category = request.args.get('category')
            return jsonify(self.api.get_plugin_list(category))
        
        @self.app.route('/api/evaluate', methods=['POST'])
        def run_evaluation():
            # 评估接口，调用EvaluationAPI
            from api.evaluation_api import EvaluationAPI
            eval_api = EvaluationAPI(self.api.engine)
            
            data = request.get_json()
            config = data.get('config', {})
            
            result = eval_api.run_evaluation(config)
            return jsonify(result)
    
    def start(self, debug: bool = False):
        """
        启动服务器
        
        Args:
            debug: 是否开启调试模式
        """
        # 初始化API
        if not self.api.initialize():
            logger.error("API初始化失败，服务器无法启动")
            return False
        
        logger.info(f"启动API服务器: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
        return True
    
    def stop(self):
        """停止服务器"""
        logger.info("停止API服务器...")
        self.api.shutdown()

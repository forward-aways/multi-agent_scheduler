"""
自动化评估API模块
提供多轮仿真实验和性能评估功能
"""

from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json
import csv
import logging
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict

from core.scheduler_engine import SchedulerEngine
from core.plugin_interface import StrategyPlugin, EnvironmentPlugin

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    # 任务完成指标
    task_completion_rate: float = 0.0  # 任务完成率
    avg_completion_time: float = 0.0  # 平均完成时间
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # 延迟指标
    avg_delay: float = 0.0  # 平均延迟
    max_delay: float = 0.0  # 最大延迟
    min_delay: float = 0.0  # 最小延迟
    delay_std: float = 0.0  # 延迟标准差
    
    # 负载均衡指标
    load_balance_score: float = 0.0  # 负载均衡度
    agent_utilization: Dict[str, float] = None  # 各智能体利用率
    
    # 资源利用指标
    avg_resource_usage: float = 0.0  # 平均资源利用率
    resource_efficiency: float = 0.0  # 资源效率
    
    # 奖励指标
    total_reward: float = 0.0  # 总奖励
    avg_reward_per_step: float = 0.0  # 每步平均奖励
    reward_variance: float = 0.0  # 奖励方差
    
    # 稳定性指标
    success_rate: float = 0.0  # 成功率
    convergence_episode: int = -1  # 收敛回合
    stability_score: float = 0.0  # 稳定性得分
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class EvaluationAPI:
    """
    评估API类
    
    功能：
    1. 自动执行多轮仿真实验
    2. 收集性能数据
    3. 生成结构化评估报告
    4. 支持结果导出
    """
    
    def __init__(self, scheduler_engine: SchedulerEngine = None):
        """
        初始化评估API
        
        Args:
            scheduler_engine: 调度引擎实例
        """
        self.engine = scheduler_engine or SchedulerEngine()
        self.evaluation_history: List[Dict[str, Any]] = []
        self.current_evaluation: Optional[Dict[str, Any]] = None
        
        # 评估回调
        self.on_episode_end: Optional[Callable] = None
        self.on_evaluation_complete: Optional[Callable] = None
        
        logger.info("评估API初始化完成")
    
    def run_evaluation(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        运行评估
        
        Args:
            config: 评估配置
                {
                    'name': '评估名称',
                    'environment': '环境插件名',
                    'strategy': '策略插件名',
                    'episodes': 100,  # 评估回合数
                    'max_steps': 1000,  # 每回合最大步数
                    'metrics': ['completion_rate', 'delay', 'load_balance'],  # 要收集的指标
                    'export_format': 'json',  # 导出格式: json, csv
                    'export_path': 'results/eval_001'  # 导出路径
                }
        
        Returns:
            评估结果
        """
        logger.info(f"开始评估: {config.get('name', 'Unnamed Evaluation')}")
        
        try:
            # 初始化引擎
            if not self.engine.initialize():
                return {
                    'success': False,
                    'message': '引擎初始化失败',
                    'timestamp': datetime.now().isoformat()
                }
            
            # 加载场景
            scenario_config = {
                'name': config.get('name', 'evaluation'),
                'environment': config.get('environment'),
                'strategy': config.get('strategy'),
                'env_config': config.get('env_config', {}),
                'strategy_config': config.get('strategy_config', {})
            }
            
            if not self.engine.load_scenario(scenario_config):
                return {
                    'success': False,
                    'message': '场景加载失败',
                    'timestamp': datetime.now().isoformat()
                }
            
            # 运行评估
            episodes = config.get('episodes', 100)
            max_steps = config.get('max_steps', 1000)
            
            evaluation_data = {
                'config': config,
                'episodes': [],
                'start_time': datetime.now().isoformat()
            }
            
            # 注册回调以收集数据
            episode_data_list = []
            
            def on_episode_end(data):
                episode_data_list.append(data)
                if self.on_episode_end:
                    self.on_episode_end(data)
            
            self.engine.register_callback('on_episode_end', on_episode_end)
            
            # 运行多个回合
            for episode in range(episodes):
                result = self.engine.run_episode(max_steps=max_steps)
                
                if not result.get('success'):
                    logger.warning(f"回合 {episode} 运行失败")
                    continue
                
                logger.info(f"完成回合 {episode+1}/{episodes}")
            
            # 计算指标
            metrics = self._calculate_metrics(episode_data_list, config)
            
            # 完成评估
            evaluation_data['end_time'] = datetime.now().isoformat()
            evaluation_data['metrics'] = metrics.to_dict()
            evaluation_data['raw_data'] = episode_data_list
            
            self.current_evaluation = evaluation_data
            self.evaluation_history.append(evaluation_data)
            
            # 导出结果
            export_result = self._export_results(evaluation_data, config)
            
            # 触发完成回调
            if self.on_evaluation_complete:
                self.on_evaluation_complete(evaluation_data)
            
            logger.info("评估完成")
            
            return {
                'success': True,
                'evaluation': {
                    'name': config.get('name'),
                    'metrics': metrics.to_dict(),
                    'episodes': episodes,
                    'duration': self._calculate_duration(
                        evaluation_data['start_time'],
                        evaluation_data['end_time']
                    ),
                    'raw_data': episode_data_list  # 添加回合原始数据
                },
                'export': export_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"评估过程中出错: {e}")
            return {
                'success': False,
                'message': f'评估失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def run_batch_evaluation(
        self,
        configs: List[Dict[str, Any]],
        comparison_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        批量评估多个配置
        
        Args:
            configs: 评估配置列表
            comparison_metrics: 用于比较的指标
        
        Returns:
            批量评估结果
        """
        logger.info(f"开始批量评估: {len(configs)} 个配置")
        
        results = []
        
        for i, config in enumerate(configs):
            logger.info(f"评估配置 {i+1}/{len(configs)}: {config.get('name', f'Config_{i}')}")
            result = self.run_evaluation(config)
            results.append(result)
        
        # 生成对比报告
        comparison = self._generate_comparison(results, comparison_metrics)
        
        return {
            'success': True,
            'total_evaluations': len(configs),
            'results': results,
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_strategies(
        self,
        environment: str,
        strategies: List[str],
        episodes: int = 50,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        对比多个策略
        
        Args:
            environment: 环境名称
            strategies: 策略名称列表
            episodes: 每个策略的评估回合数
            metrics: 评估指标
        
        Returns:
            对比结果
        """
        logger.info(f"开始策略对比: {len(strategies)} 个策略")
        
        configs = []
        for strategy in strategies:
            config = {
                'name': f'{strategy}_vs_others',
                'environment': environment,
                'strategy': strategy,
                'episodes': episodes,
                'metrics': metrics or ['completion_rate', 'delay', 'load_balance', 'reward']
            }
            configs.append(config)
        
        return self.run_batch_evaluation(configs, metrics)
    
    def _calculate_metrics(
        self,
        episode_data_list: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> EvaluationMetrics:
        """
        计算评估指标
        
        Args:
            episode_data_list: 回合数据列表
            config: 评估配置
        
        Returns:
            评估指标
        """
        metrics = EvaluationMetrics()
        
        if not episode_data_list:
            return metrics
        
        # 基本统计
        total_rewards = [ep['total_reward'] for ep in episode_data_list]
        steps = [ep['steps'] for ep in episode_data_list]
        
        metrics.total_reward = np.sum(total_rewards)
        metrics.avg_reward_per_step = np.mean(total_rewards) / max(1, np.mean(steps))
        metrics.reward_variance = np.var(total_rewards)
        
        # 成功率
        metrics.success_rate = len(episode_data_list) / max(1, config.get('episodes', 100))
        
        # 延迟指标（如果有延迟数据）
        delays = []
        for ep in episode_data_list:
            if 'delays' in ep:
                delays.extend(ep['delays'])
        
        if delays:
            metrics.avg_delay = np.mean(delays)
            metrics.max_delay = np.max(delays)
            metrics.min_delay = np.min(delays)
            metrics.delay_std = np.std(delays)
        
        # 负载均衡（如果有利用率数据）
        utilizations = []
        for ep in episode_data_list:
            if 'agent_utilization' in ep:
                utilizations.append(ep['agent_utilization'])
        
        if utilizations:
            # 计算负载均衡度（使用变异系数的倒数）
            avg_utils = [np.mean(list(u.values())) for u in utilizations]
            metrics.load_balance_score = 1.0 / (1.0 + np.std(avg_utils))
        
        # 任务完成率（如果有任务数据）
        total_tasks = 0
        completed_tasks = 0
        for ep in episode_data_list:
            if 'tasks' in ep:
                total_tasks += ep['tasks'].get('total', 0)
                completed_tasks += ep['tasks'].get('completed', 0)
        
        if total_tasks > 0:
            metrics.total_tasks = total_tasks
            metrics.completed_tasks = completed_tasks
            metrics.failed_tasks = total_tasks - completed_tasks
            metrics.task_completion_rate = completed_tasks / total_tasks
        
        # 收敛性检测
        window_size = min(10, len(total_rewards) // 4)
        if window_size > 0:
            for i in range(window_size, len(total_rewards)):
                recent_avg = np.mean(total_rewards[i-window_size:i])
                overall_avg = np.mean(total_rewards[:i])
                if abs(recent_avg - overall_avg) < 0.05 * abs(overall_avg):
                    metrics.convergence_episode = i
                    break
        
        # 稳定性得分
        if len(total_rewards) > 1:
            reward_changes = np.diff(total_rewards)
            metrics.stability_score = 1.0 / (1.0 + np.std(reward_changes))
        
        return metrics
    
    def _export_results(
        self,
        evaluation_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        导出评估结果
        
        Args:
            evaluation_data: 评估数据
            config: 配置
        
        Returns:
            导出结果
        """
        export_format = config.get('export_format', 'json')
        export_path = config.get('export_path')
        
        if not export_path:
            return {'success': False, 'message': '未指定导出路径'}
        
        # 创建导出目录
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if export_format == 'json':
                # JSON导出
                json_file = export_dir / 'evaluation_results.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_data, f, indent=2, ensure_ascii=False, default=str)
                
                # 同时导出一个简化的报告
                report_file = export_dir / 'evaluation_report.json'
                report = {
                    'name': config.get('name'),
                    'config': config,
                    'metrics': evaluation_data.get('metrics', {}),
                    'summary': {
                        'total_episodes': len(evaluation_data.get('episodes', [])),
                        'start_time': evaluation_data.get('start_time'),
                        'end_time': evaluation_data.get('end_time')
                    }
                }
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                return {
                    'success': True,
                    'format': 'json',
                    'files': [str(json_file), str(report_file)]
                }
            
            elif export_format == 'csv':
                # CSV导出
                csv_file = export_dir / 'episode_data.csv'
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    if evaluation_data.get('raw_data'):
                        writer = None
                        for episode_data in evaluation_data['raw_data']:
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=episode_data.keys())
                                writer.writeheader()
                            writer.writerow(episode_data)
                
                return {
                    'success': True,
                    'format': 'csv',
                    'files': [str(csv_file)]
                }
            
            else:
                return {'success': False, 'message': f'不支持的导出格式: {export_format}'}
                
        except Exception as e:
            logger.error(f"导出结果时出错: {e}")
            return {'success': False, 'message': str(e)}
    
    def _generate_comparison(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        生成对比报告
        
        Args:
            results: 评估结果列表
            metrics: 对比指标
        
        Returns:
            对比报告
        """
        if not results:
            return {}
        
        metrics = metrics or ['completion_rate', 'delay', 'load_balance', 'reward']
        comparison = {
            'metrics_compared': metrics,
            'strategies': [],
            'rankings': {}
        }
        
        # 收集各策略的指标
        for result in results:
            if not result.get('success'):
                continue
            
            eval_data = result.get('evaluation', {})
            strategy_name = eval_data.get('name', 'Unknown')
            strategy_metrics = eval_data.get('metrics', {})
            
            comparison['strategies'].append({
                'name': strategy_name,
                'metrics': strategy_metrics
            })
        
        # 生成排名
        for metric in metrics:
            ranked = sorted(
                comparison['strategies'],
                key=lambda x: x['metrics'].get(metric, 0),
                reverse=True
            )
            comparison['rankings'][metric] = [
                {'name': s['name'], 'value': s['metrics'].get(metric, 0)}
                for s in ranked
            ]
        
        # 确定最佳策略
        if comparison['strategies']:
            # 简单投票机制
            votes = {}
            for metric, ranking in comparison['rankings'].items():
                if ranking:
                    best = ranking[0]['name']
                    votes[best] = votes.get(best, 0) + 1
            
            if votes:
                best_strategy = max(votes, key=votes.get)
                comparison['best_strategy'] = best_strategy
                comparison['voting_results'] = votes
        
        return comparison
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """
        计算持续时间
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            持续时间（秒）
        """
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            return (end - start).total_seconds()
        except:
            return 0.0
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        获取评估历史
        
        Returns:
            评估历史列表
        """
        return self.evaluation_history
    
    def get_evaluation_report(self, evaluation_index: int = -1) -> Optional[Dict[str, Any]]:
        """
        获取评估报告
        
        Args:
            evaluation_index: 评估索引（-1表示最新的）
        
        Returns:
            评估报告
        """
        if not self.evaluation_history:
            return None
        
        try:
            return self.evaluation_history[evaluation_index]
        except IndexError:
            return None
    
    def generate_summary_report(self, output_path: str = None) -> Dict[str, Any]:
        """
        生成汇总报告
        
        Args:
            output_path: 输出路径
        
        Returns:
            汇总报告
        """
        if not self.evaluation_history:
            return {'success': False, 'message': '没有评估历史'}
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'evaluations': [],
            'overall_statistics': {}
        }
        
        # 收集所有评估的基本信息
        all_completion_rates = []
        all_rewards = []
        all_delays = []
        
        for eval_data in self.evaluation_history:
            metrics = eval_data.get('metrics', {})
            
            summary['evaluations'].append({
                'name': eval_data['config'].get('name'),
                'environment': eval_data['config'].get('environment'),
                'strategy': eval_data['config'].get('strategy'),
                'metrics': metrics
            })
            
            all_completion_rates.append(metrics.get('task_completion_rate', 0))
            all_rewards.append(metrics.get('total_reward', 0))
            all_delays.append(metrics.get('avg_delay', 0))
        
        # 计算总体统计
        summary['overall_statistics'] = {
            'avg_completion_rate': np.mean(all_completion_rates) if all_completion_rates else 0,
            'avg_total_reward': np.mean(all_rewards) if all_rewards else 0,
            'avg_delay': np.mean(all_delays) if all_delays else 0,
            'best_completion_rate': max(all_completion_rates) if all_completion_rates else 0,
            'best_total_reward': max(all_rewards) if all_rewards else 0
        }
        
        # 导出报告
        if output_path:
            output_file = Path(output_path) / 'summary_report.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            summary['export_path'] = str(output_file)
        
        return summary

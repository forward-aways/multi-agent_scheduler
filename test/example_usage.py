"""
插件系统使用示例
展示如何使用插件式调度引擎和API模块
"""

import sys
sys.path.insert(0, 'd:\\graduation_project\\multi_agent_scheduler')

from core.scheduler_engine import SchedulerEngine
from api.scheduler_api import SchedulerAPI, APIServer
from api.evaluation_api import EvaluationAPI


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "="*60)
    print("示例1: 基本使用 - 创建引擎并加载场景")
    print("="*60)
    
    # 创建调度引擎
    engine = SchedulerEngine()
    
    # 初始化引擎（自动发现并加载插件）
    engine.initialize()
    
    # 加载场景配置
    scenario_config = {
        'name': 'server_scheduling_demo',
        'environment': 'server_environment',
        'strategy': 'random_strategy',
        'env_config': {
            'num_servers': 3,
            'max_steps': 50
        },
        'strategy_config': {
            'num_actions': 5
        }
    }
    
    # 加载场景
    success = engine.load_scenario(scenario_config)
    print(f"场景加载: {'成功' if success else '失败'}")
    
    # 运行一个回合
    result = engine.run_episode(max_steps=20)
    print(f"回合运行: {'成功' if result['success'] else '失败'}")
    
    if result['success']:
        print(f"  - 步数: {result['episode_data']['steps']}")
        print(f"  - 总奖励: {result['episode_data']['total_reward']:.2f}")
    
    # 关闭引擎
    engine.shutdown()


def example_2_strategy_switching():
    """示例2: 运行时策略切换"""
    print("\n" + "="*60)
    print("示例2: 运行时策略切换")
    print("="*60)
    
    engine = SchedulerEngine()
    engine.initialize()
    
    # 加载初始场景
    engine.load_scenario({
        'name': 'strategy_switch_demo',
        'environment': 'server_environment',
        'strategy': 'random_strategy',
        'env_config': {'num_servers': 3, 'max_steps': 50},
        'strategy_config': {'num_actions': 5}
    })
    
    # 使用随机策略运行
    print("\n使用随机策略:")
    result1 = engine.run_episode(max_steps=10)
    print(f"  奖励: {result1['episode_data']['total_reward']:.2f}")
    
    # 切换到MAPPO策略
    print("\n切换到MAPPO策略...")
    engine.switch_strategy('mappo_strategy', {'num_actions': 5})
    
    # 使用MAPPO策略运行
    print("使用MAPPO策略:")
    result2 = engine.run_episode(max_steps=10)
    print(f"  奖励: {result2['episode_data']['total_reward']:.2f}")
    
    engine.shutdown()


def example_3_api_usage():
    """示例3: 使用API进行任务分配"""
    print("\n" + "="*60)
    print("示例3: 使用API进行任务分配")
    print("="*60)
    
    # 创建API实例
    api = SchedulerAPI()
    api.initialize()
    
    # 加载场景
    api.load_scenario({
        'name': 'api_demo',
        'environment': 'server_environment',
        'strategy': 'random_strategy',
        'env_config': {'num_servers': 3, 'max_steps': 50},
        'strategy_config': {'num_actions': 5}
    })
    
    # 请求任务分配
    task = {
        'task_id': 'task_001',
        'task_type': 'computation',
        'requirements': {'cpu': 10, 'memory': 20},
        'priority': 5
    }
    
    result = api.request_task_allocation(task, agent_id='server_0')
    print(f"\n任务分配请求:")
    print(f"  任务ID: {task['task_id']}")
    print(f"  分配结果: {result}")
    
    # 批量任务分配
    tasks = [
        {'task_id': f'task_{i:03d}', 'task_type': 'computation', 'priority': i}
        for i in range(1, 6)
    ]
    
    batch_result = api.batch_allocate_tasks(tasks)
    print(f"\n批量任务分配:")
    print(f"  总任务数: {batch_result['total_tasks']}")
    print(f"  成功分配: {batch_result['successful_allocations']}")
    print(f"  失败分配: {batch_result['failed_allocations']}")
    
    api.shutdown()


def example_4_evaluation():
    """示例4: 自动化评估"""
    print("\n" + "="*60)
    print("示例4: 自动化评估")
    print("="*60)
    
    # 创建评估API
    eval_api = EvaluationAPI()
    
    # 配置评估
    config = {
        'name': 'random_vs_mappo',
        'environment': 'server_environment',
        'strategy': 'random_strategy',
        'episodes': 5,
        'max_steps': 30,
        'env_config': {
            'num_servers': 3,
            'max_steps': 30
        },
        'strategy_config': {
            'num_actions': 5
        },
        'export_format': 'json',
        'export_path': 'evaluations/example_run'
    }
    
    # 运行评估
    print(f"\n运行评估: {config['name']}")
    print(f"  回合数: {config['episodes']}")
    
    result = eval_api.run_evaluation(config)
    
    if result['success']:
        evaluation = result['evaluation']
        metrics = evaluation['metrics']
        
        print(f"\n评估结果:")
        print(f"  成功率: {metrics.get('success_rate', 0):.2%}")
        print(f"  总奖励: {metrics.get('total_reward', 0):.2f}")
        print(f"  平均奖励/步: {metrics.get('avg_reward_per_step', 0):.4f}")
        print(f"  稳定性得分: {metrics.get('stability_score', 0):.4f}")
        
        if result['export']['success']:
            print(f"\n结果已导出到: {result['export']['files']}")
    
    # 生成汇总报告
    summary = eval_api.generate_summary_report('evaluations')
    print(f"\n汇总报告已生成: {summary.get('export_path', 'N/A')}")


def example_5_batch_evaluation():
    """示例5: 批量评估与对比"""
    print("\n" + "="*60)
    print("示例5: 批量评估与对比")
    print("="*60)
    
    eval_api = EvaluationAPI()
    
    # 配置多个场景进行对比
    configs = [
        {
            'name': 'random_baseline',
            'environment': 'server_environment',
            'strategy': 'random_strategy',
            'episodes': 3,
            'max_steps': 20,
            'env_config': {'num_servers': 3, 'max_steps': 20},
            'strategy_config': {'num_actions': 5}
        },
        {
            'name': 'mappo_test',
            'environment': 'server_environment',
            'strategy': 'mappo_strategy',
            'episodes': 3,
            'max_steps': 20,
            'env_config': {'num_servers': 3, 'max_steps': 20},
            'strategy_config': {'num_actions': 5}
        }
    ]
    
    print(f"\n批量评估 {len(configs)} 个配置...")
    
    # 运行批量评估
    result = eval_api.run_batch_evaluation(
        configs,
        comparison_metrics=['success_rate', 'total_reward', 'stability_score']
    )
    
    if result['success']:
        print(f"\n批量评估完成!")
        print(f"  总评估数: {result['total_evaluations']}")
        
        comparison = result['comparison']
        if 'best_strategy' in comparison:
            print(f"\n对比结果:")
            print(f"  最佳策略: {comparison['best_strategy']}")
            print(f"  投票结果: {comparison['voting_results']}")
            
            print(f"\n各指标排名:")
            for metric, ranking in comparison['rankings'].items():
                print(f"  {metric}:")
                for i, item in enumerate(ranking[:3], 1):
                    print(f"    {i}. {item['name']}: {item['value']:.4f}")


def example_6_plugin_management():
    """示例6: 插件管理"""
    print("\n" + "="*60)
    print("示例6: 插件管理")
    print("="*60)
    
    from core.plugin_manager import PluginManager
    
    # 创建插件管理器
    pm = PluginManager(plugin_dirs=['plugins'])
    
    # 发现插件
    discovered = pm.discover_plugins()
    print(f"\n发现 {len(discovered)} 个插件:")
    for name in discovered:
        print(f"  - {name}")
    
    # 加载所有插件
    print(f"\n加载插件...")
    for name in discovered:
        success = pm.load_plugin(name)
        print(f"  {name}: {'✓' if success else '✗'}")
    
    # 列出所有插件
    plugins = pm.list_plugins()
    print(f"\n已加载插件详情:")
    for plugin in plugins:
        print(f"  - {plugin['name']} v{plugin['version']} ({plugin['category']})")
        if plugin.get('is_active_plugin'):
            print(f"    [当前激活]")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("多智能体调度系统 - 插件系统使用示例")
    print("="*60)
    
    try:
        # 运行示例
        example_1_basic_usage()
        example_2_strategy_switching()
        example_3_api_usage()
        example_4_evaluation()
        example_5_batch_evaluation()
        example_6_plugin_management()
        
        print("\n" + "="*60)
        print("所有示例运行完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

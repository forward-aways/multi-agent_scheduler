"""
插件系统测试脚本
测试插件式调度引擎和API模块的功能
"""

import sys
sys.path.insert(0, 'd:\\graduation_project\\multi_agent_scheduler')

import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_plugin_manager():
    """测试插件管理器"""
    print("\n" + "="*60)
    print("测试1: 插件管理器")
    print("="*60)
    
    from core.plugin_manager import PluginManager
    
    # 创建插件管理器
    pm = PluginManager(plugin_dirs=['plugins'])
    
    # 发现插件
    discovered = pm.discover_plugins()
    print(f"\n发现插件: {discovered}")
    
    # 加载插件
    for plugin_name in discovered:
        success = pm.load_plugin(plugin_name)
        print(f"加载插件 {plugin_name}: {'成功' if success else '失败'}")
    
    # 列出所有插件
    plugins = pm.list_plugins()
    print(f"\n已加载插件数量: {len(plugins)}")
    for plugin in plugins:
        print(f"  - {plugin['name']} (v{plugin['version']}, {plugin['category']})")
    
    return pm


def test_scheduler_engine(pm):
    """测试调度引擎"""
    print("\n" + "="*60)
    print("测试2: 调度引擎")
    print("="*60)
    
    from core.scheduler_engine import SchedulerEngine
    
    # 创建引擎
    engine = SchedulerEngine(config={'plugin_dirs': ['plugins']})
    engine.plugin_manager = pm
    
    # 初始化
    success = engine.initialize()
    print(f"\n引擎初始化: {'成功' if success else '失败'}")
    
    # 获取状态
    status = engine.get_status()
    print(f"\n引擎状态:")
    print(f"  - 运行中: {status['is_running']}")
    print(f"  - 已加载插件: {status['loaded_plugins']}")
    print(f"  - 激活插件: {status['active_plugins']}")
    
    return engine


def test_scenario_loading(engine):
    """测试场景加载"""
    print("\n" + "="*60)
    print("测试3: 场景加载")
    print("="*60)
    
    # 加载服务器调度场景
    scenario_config = {
        'name': 'server_scheduling_test',
        'environment': 'server_environment',
        'strategy': 'random_strategy',
        'env_config': {
            'num_servers': 3,
            'num_tasks': 10,
            'max_steps': 50
        },
        'strategy_config': {
            'num_actions': 5
        }
    }
    
    success = engine.load_scenario(scenario_config)
    print(f"\n加载场景: {'成功' if success else '失败'}")
    
    if success:
        print(f"  - 环境: {engine.current_environment.name if engine.current_environment else 'None'}")
        print(f"  - 策略: {engine.current_strategy.name if engine.current_strategy else 'None'}")
    
    return success


def test_strategy_switching(engine):
    """测试策略切换"""
    print("\n" + "="*60)
    print("测试4: 策略切换")
    print("="*60)
    
    # 切换到MAPPO策略（如果存在）
    success = engine.switch_strategy('mappo_strategy', {'num_actions': 5})
    print(f"\n切换到MAPPO策略: {'成功' if success else '失败'}")
    
    if success:
        print(f"  - 当前策略: {engine.current_strategy.name}")
    
    # 切回随机策略
    success = engine.switch_strategy('random_strategy', {'num_actions': 5})
    print(f"\n切回随机策略: {'成功' if success else '失败'}")
    
    return success


def test_api_module():
    """测试API模块"""
    print("\n" + "="*60)
    print("测试5: API模块")
    print("="*60)
    
    from api.scheduler_api import SchedulerAPI
    
    # 创建API
    api = SchedulerAPI()
    
    # 初始化
    success = api.initialize()
    print(f"\nAPI初始化: {'成功' if success else '失败'}")
    
    if success:
        # 获取状态
        status = api.get_engine_status()
        print(f"\n引擎状态: {status}")
        
        # 获取可用场景
        scenarios = api.get_available_scenarios()
        print(f"\n可用环境: {len(scenarios.get('environments', []))} 个")
        print(f"可用策略: {len(scenarios.get('strategies', []))} 个")
    
    return api


def test_evaluation_api():
    """测试评估API"""
    print("\n" + "="*60)
    print("测试6: 评估API")
    print("="*60)
    
    from api.evaluation_api import EvaluationAPI
    
    # 创建评估API
    eval_api = EvaluationAPI()
    
    # 配置评估
    config = {
        'name': 'test_evaluation',
        'environment': 'server_environment',
        'strategy': 'random_strategy',
        'episodes': 3,
        'max_steps': 20,
        'env_config': {
            'num_servers': 3,
            'num_tasks': 5,
            'max_steps': 20
        },
        'strategy_config': {
            'num_actions': 5
        },
        'export_format': 'json',
        'export_path': 'evaluations/test_run'
    }
    
    print(f"\n运行评估: {config['name']}")
    print(f"  - 回合数: {config['episodes']}")
    print(f"  - 每回合最大步数: {config['max_steps']}")
    
    # 运行评估
    result = eval_api.run_evaluation(config)
    
    if result.get('success'):
        print(f"\n评估成功!")
        evaluation = result.get('evaluation', {})
        metrics = evaluation.get('metrics', {})
        
        print(f"\n评估指标:")
        print(f"  - 成功率: {metrics.get('success_rate', 0):.2%}")
        print(f"  - 总奖励: {metrics.get('total_reward', 0):.2f}")
        print(f"  - 平均奖励/步: {metrics.get('avg_reward_per_step', 0):.4f}")
        
        export = result.get('export', {})
        if export.get('success'):
            print(f"\n导出结果:")
            for file in export.get('files', []):
                print(f"  - {file}")
    else:
        print(f"\n评估失败: {result.get('message')}")
    
    return eval_api


def test_batch_evaluation():
    """测试批量评估"""
    print("\n" + "="*60)
    print("测试7: 批量评估")
    print("="*60)
    
    from api.evaluation_api import EvaluationAPI
    
    # 创建评估API
    eval_api = EvaluationAPI()
    
    # 配置多个场景
    configs = [
        {
            'name': 'random_strategy_test',
            'environment': 'server_environment',
            'strategy': 'random_strategy',
            'episodes': 2,
            'max_steps': 10,
            'env_config': {'num_servers': 3, 'num_tasks': 5, 'max_steps': 10},
            'strategy_config': {'num_actions': 5}
        },
        {
            'name': 'mappo_strategy_test',
            'environment': 'server_environment',
            'strategy': 'mappo_strategy',
            'episodes': 2,
            'max_steps': 10,
            'env_config': {'num_servers': 3, 'num_tasks': 5, 'max_steps': 10},
            'strategy_config': {'num_actions': 5}
        }
    ]
    
    print(f"\n批量评估 {len(configs)} 个配置")
    
    # 运行批量评估
    result = eval_api.run_batch_evaluation(configs)
    
    if result.get('success'):
        print(f"\n批量评估成功!")
        print(f"  - 总评估数: {result.get('total_evaluations')}")
        
        comparison = result.get('comparison', {})
        if comparison:
            print(f"\n对比结果:")
            print(f"  - 最佳策略: {comparison.get('best_strategy', 'N/A')}")
            print(f"  - 投票结果: {comparison.get('voting_results', {})}")
    else:
        print(f"\n批量评估失败: {result.get('message')}")
    
    return result


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("多智能体调度系统 - 插件系统测试")
    print("="*60)
    
    try:
        # 测试1: 插件管理器
        pm = test_plugin_manager()
        
        # 测试2: 调度引擎
        engine = test_scheduler_engine(pm)
        
        # 测试3: 场景加载
        test_scenario_loading(engine)
        
        # 测试4: 策略切换
        test_strategy_switching(engine)
        
        # 测试5: API模块
        api = test_api_module()
        
        # 测试6: 评估API
        eval_api = test_evaluation_api()
        
        # 测试7: 批量评估
        test_batch_evaluation()
        
        print("\n" + "="*60)
        print("所有测试完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

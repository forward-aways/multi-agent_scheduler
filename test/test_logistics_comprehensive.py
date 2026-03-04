"""
物流调度全面测试 - 覆盖各种场景
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv

class LogisticsTestSuite:
    """物流调度测试套件"""
    
    def __init__(self):
        self.test_results = []
    
    def run_all_tests(self):
        """运行所有测试"""
        print("="*80)
        print("物流调度全面测试")
        print("="*80)
        
        # 基础功能测试
        self.test_basic_order_lifecycle()
        self.test_multiple_orders_same_warehouse()
        self.test_multiple_vehicles_collaboration()
        self.test_order_priority()
        self.test_insufficient_inventory()
        
        # 边界情况测试
        self.test_empty_order_list()
        self.test_vehicle_capacity_limit()
        self.test_warehouse_capacity_limit()
        self.test_long_distance_delivery()
        
        # 并发测试
        self.test_simultaneous_actions()
        self.test_rapid_order_generation()
        
        # 生成报告
        self.generate_report()
    
    def test_basic_order_lifecycle(self):
        """测试1: 基础订单生命周期"""
        print("\n" + "="*80)
        print("测试1: 基础订单生命周期")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 1,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 50,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 添加订单
        order_pos = np.array([50.0, 50.0])
        env.pending_orders.append([order_pos, 10, 5])
        env.all_orders.append({
            'position': order_pos,
            'quantity': 10,
            'priority': 5,
            'status': 'pending'
        })
        
        print(f"初始: 订单状态={env.all_orders[0]['status']}")
        
        # 步骤1: 仓库分配订单
        actions = {'warehouse_0': 0, 'vehicle_0': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        status_after_assign = env.all_orders[0]['status']
        print(f"仓库分配后: 订单状态={status_after_assign}")
        
        # 步骤2-3: 车辆去仓库
        for _ in range(2):
            actions = {'warehouse_0': 3, 'vehicle_0': 0}
            obs, rewards, done, truncated, info = env.step(actions)
        print(f"车辆到仓库后: 载货={env.vehicle_cargo[0]}")
        
        # 步骤4-10: 车辆去配送
        for _ in range(10):
            actions = {'warehouse_0': 3, 'vehicle_0': 1}
            obs, rewards, done, truncated, info = env.step(actions)
            if env.vehicle_cargo[0] == 0:
                break
        
        status_final = env.all_orders[0]['status']
        print(f"配送完成后: 订单状态={status_final}, 已完成={env.completed_orders}")
        
        passed = (status_after_assign == 'delivering' and 
                 status_final == 'completed' and 
                 env.completed_orders == 1)
        
        self.test_results.append(("基础订单生命周期", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_multiple_orders_same_warehouse(self):
        """测试2: 同一仓库多个订单"""
        print("\n" + "="*80)
        print("测试2: 同一仓库多个订单")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 2,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 100,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 添加3个订单
        orders = [
            [np.array([30.0, 30.0]), 5, 5],
            [np.array([60.0, 60.0]), 5, 3],
            [np.array([80.0, 20.0]), 5, 4]
        ]
        
        for i, order in enumerate(orders):
            env.pending_orders.append(order)
            env.all_orders.append({
                'position': order[0],
                'quantity': order[1],
                'priority': order[2],
                'status': 'pending'
            })
        
        print(f"初始: 3个待处理订单")
        
        # 分配所有订单
        for step in range(5):
            actions = {'warehouse_0': 0, 'vehicle_0': 3, 'vehicle_1': 3}
            obs, rewards, done, truncated, info = env.step(actions)
            if len(env.pending_orders) == 0:
                break
        
        print(f"分配后: pending={len(env.pending_orders)}, delivering={len(env.delivering_orders)}")
        print(f"仓库订单数: {len(env.warehouse_orders[0])}")
        
        # 两辆车都去装货配送
        for step in range(30):
            actions = {'warehouse_0': 3, 'vehicle_0': 0, 'vehicle_1': 0}
            obs, rewards, done, truncated, info = env.step(actions)
            
            # 装货后去配送
            if env.vehicle_cargo[0] > 0:
                actions['vehicle_0'] = 1
            if env.vehicle_cargo[1] > 0:
                actions['vehicle_1'] = 1
            
            obs, rewards, done, truncated, info = env.step(actions)
            
            if env.completed_orders >= 3:
                break
        
        print(f"最终: 已完成={env.completed_orders}")
        print(f"订单状态: {[o['status'] for o in env.all_orders]}")
        
        passed = env.completed_orders == 3
        self.test_results.append(("同一仓库多个订单", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_multiple_vehicles_collaboration(self):
        """测试3: 多车辆协作"""
        print("\n" + "="*80)
        print("测试3: 多车辆协作")
        print("="*80)
        
        env_config = {
            'num_warehouses': 2,
            'num_vehicles': 3,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 100,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 仓库0的订单
        env.pending_orders.append([np.array([20.0, 40.0]), 5, 5])
        env.all_orders.append({
            'position': np.array([20.0, 40.0]),
            'quantity': 5,
            'priority': 5,
            'status': 'pending'
        })
        
        # 仓库1的订单
        env.pending_orders.append([np.array([60.0, 60.0]), 5, 3])
        env.all_orders.append({
            'position': np.array([60.0, 60.0]),
            'quantity': 5,
            'priority': 3,
            'status': 'pending'
        })
        
        print(f"初始: 仓库0订单+仓库1订单")
        
        # 分配订单
        actions = {'warehouse_0': 0, 'warehouse_1': 0, 
                  'vehicle_0': 3, 'vehicle_1': 3, 'vehicle_2': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        
        print(f"分配后: warehouse0={len(env.warehouse_orders[0])}, warehouse1={len(env.warehouse_orders[1])}")
        
        # 车辆协作配送
        for step in range(50):
            actions = {'warehouse_0': 3, 'warehouse_1': 3,
                      'vehicle_0': 0, 'vehicle_1': 0, 'vehicle_2': 0}
            
            obs, rewards, done, truncated, info = env.step(actions)
            
            # 装货后去配送
            for i in range(3):
                if env.vehicle_cargo[i] > 0:
                    actions[f'vehicle_{i}'] = 1
            
            obs, rewards, done, truncated, info = env.step(actions)
            
            if env.completed_orders >= 2:
                break
        
        print(f"最终: 已完成={env.completed_orders}")
        print(f"车辆载货历史: 车辆0曾经载货={env.vehicle_cargo[0] > 0}, 车辆1曾经载货={env.vehicle_cargo[1] > 0}, 车辆2曾经载货={env.vehicle_cargo[2] > 0}")
        
        passed = env.completed_orders == 2
        self.test_results.append(("多车辆协作", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_order_priority(self):
        """测试4: 订单优先级"""
        print("\n" + "="*80)
        print("测试4: 订单优先级")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 1,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 50,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 添加低优先级订单
        env.pending_orders.append([np.array([50.0, 50.0]), 5, 1])
        env.all_orders.append({
            'position': np.array([50.0, 50.0]),
            'quantity': 5,
            'priority': 1,
            'status': 'pending'
        })
        
        # 添加高优先级订单
        env.pending_orders.append([np.array([60.0, 60.0]), 5, 10])
        env.all_orders.append({
            'position': np.array([60.0, 60.0]),
            'quantity': 5,
            'priority': 10,
            'status': 'pending'
        })
        
        # 分配订单 - 应该优先分配高优先级
        actions = {'warehouse_0': 0, 'vehicle_0': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 检查哪个订单被分配了
        assigned_order = env.warehouse_orders[0][0] if len(env.warehouse_orders[0]) > 0 else None
        
        if assigned_order is not None:
            is_high_priority = assigned_order[2] == 10
            print(f"高优先级订单被优先分配: {is_high_priority}")
            passed = is_high_priority
        else:
            print("没有订单被分配")
            passed = False
        
        self.test_results.append(("订单优先级", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_insufficient_inventory(self):
        """测试5: 库存不足情况"""
        print("\n" + "="*80)
        print("测试5: 库存不足情况")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 1,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 50,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 设置低库存
        env.warehouse_inventory[0] = 5
        
        # 添加大订单（超过库存）
        env.pending_orders.append([np.array([50.0, 50.0]), 20, 5])
        env.all_orders.append({
            'position': np.array([50.0, 50.0]),
            'quantity': 20,
            'priority': 5,
            'status': 'pending'
        })
        
        print(f"初始库存: {env.warehouse_inventory[0]}, 订单需求: 20")
        
        # 尝试分配
        actions = {'warehouse_0': 0, 'vehicle_0': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 检查订单是否还在pending（因为库存不足不应该分配）
        order_still_pending = len(env.pending_orders) > 0
        print(f"订单因库存不足未被分配: {order_still_pending}")
        
        passed = order_still_pending
        self.test_results.append(("库存不足处理", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_empty_order_list(self):
        """测试6: 空订单列表"""
        print("\n" + "="*80)
        print("测试6: 空订单列表")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 1,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 50,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 没有订单时执行动作
        try:
            for _ in range(10):
                actions = {'warehouse_0': 0, 'vehicle_0': 0}
                obs, rewards, done, truncated, info = env.step(actions)
            
            print("空订单列表处理正常，无异常")
            passed = True
        except Exception as e:
            print(f"空订单列表处理异常: {e}")
            passed = False
        
        self.test_results.append(("空订单列表", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_vehicle_capacity_limit(self):
        """测试7: 车辆容量限制"""
        print("\n" + "="*80)
        print("测试7: 车辆容量限制")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 1,
            'warehouse_capacity': 100,
            'vehicle_capacity': 10,  # 小容量
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 50,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 添加大订单
        env.pending_orders.append([np.array([50.0, 50.0]), 15, 5])
        env.all_orders.append({
            'position': np.array([50.0, 50.0]),
            'quantity': 15,
            'priority': 5,
            'status': 'pending'
        })
        
        # 分配订单
        actions = {'warehouse_0': 0, 'vehicle_0': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 车辆去仓库装货
        for _ in range(2):
            actions = {'warehouse_0': 3, 'vehicle_0': 0}
            obs, rewards, done, truncated, info = env.step(actions)
        
        # 检查车辆载货是否超过容量
        cargo = env.vehicle_cargo[0]
        capacity = env.vehicle_capacity
        
        print(f"车辆载货: {cargo}, 容量: {capacity}")
        print(f"载货未超过容量: {cargo <= capacity}")
        
        passed = cargo <= capacity
        self.test_results.append(("车辆容量限制", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_warehouse_capacity_limit(self):
        """测试8: 仓库容量限制"""
        print("\n" + "="*80)
        print("测试8: 仓库容量限制")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 1,
            'warehouse_capacity': 50,  # 小容量
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [100.0, 100.0],
            'max_steps': 50,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        initial_inventory = env.warehouse_inventory[0]
        capacity = env.warehouse_capacity
        
        print(f"初始库存: {initial_inventory}, 容量: {capacity}")
        print(f"初始库存未超过容量: {initial_inventory <= capacity}")
        
        passed = initial_inventory <= capacity
        self.test_results.append(("仓库容量限制", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_long_distance_delivery(self):
        """测试9: 长距离配送"""
        print("\n" + "="*80)
        print("测试9: 长距离配送")
        print("="*80)
        
        env_config = {
            'num_warehouses': 1,
            'num_vehicles': 1,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 2.0,  # 慢速
            'order_generation_rate': 0,
            'max_pending_orders': 10,
            'map_size': [200.0, 200.0],  # 大地图
            'max_steps': 200,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 远距离订单
        env.pending_orders.append([np.array([150.0, 150.0]), 5, 5])
        env.all_orders.append({
            'position': np.array([150.0, 150.0]),
            'quantity': 5,
            'priority': 5,
            'status': 'pending'
        })
        
        # 分配
        actions = {'warehouse_0': 0, 'vehicle_0': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 车辆去仓库装货
        for _ in range(2):
            actions = {'warehouse_0': 3, 'vehicle_0': 0}
            obs, rewards, done, truncated, info = env.step(actions)
        
        # 长距离配送
        steps_to_deliver = 0
        for step in range(100):
            actions = {'warehouse_0': 3, 'vehicle_0': 1}
            obs, rewards, done, truncated, info = env.step(actions)
            steps_to_deliver += 1
            if env.vehicle_cargo[0] == 0:
                break
        
        print(f"配送完成用时: {steps_to_deliver}步")
        print(f"订单状态: {env.all_orders[0]['status']}")
        
        passed = env.all_orders[0]['status'] == 'completed'
        self.test_results.append(("长距离配送", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_simultaneous_actions(self):
        """测试10: 并发动作"""
        print("\n" + "="*80)
        print("测试10: 并发动作")
        print("="*80)
        
        env_config = {
            'num_warehouses': 3,
            'num_vehicles': 5,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 0,
            'max_pending_orders': 20,
            'map_size': [100.0, 100.0],
            'max_steps': 100,
            'manual_mode': True
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 添加多个订单
        for i in range(5):
            pos = np.array([20.0 + i*15, 30.0 + i*10])
            env.pending_orders.append([pos, 5, i+1])
            env.all_orders.append({
                'position': pos,
                'quantity': 5,
                'priority': i+1,
                'status': 'pending'
            })
        
        # 所有智能体同时动作
        try:
            for step in range(20):
                actions = {}
                for i in range(3):
                    actions[f'warehouse_{i}'] = 0  # 都尝试分配
                for i in range(5):
                    actions[f'vehicle_{i}'] = np.random.randint(0, 4)  # 随机动作
                
                obs, rewards, done, truncated, info = env.step(actions)
            
            print(f"并发动作执行正常")
            print(f"最终状态: 已完成={env.completed_orders}, 配送中={len(env.delivering_orders)}")
            passed = True
        except Exception as e:
            print(f"并发动作异常: {e}")
            passed = False
        
        self.test_results.append(("并发动作", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def test_rapid_order_generation(self):
        """测试11: 快速订单生成"""
        print("\n" + "="*80)
        print("测试11: 快速订单生成")
        print("="*80)
        
        env_config = {
            'num_warehouses': 2,
            'num_vehicles': 3,
            'warehouse_capacity': 100,
            'vehicle_capacity': 20,
            'vehicle_speed': 5.0,
            'order_generation_rate': 5,  # 高生成率
            'max_pending_orders': 20,
            'map_size': [100.0, 100.0],
            'max_steps': 100,
            'manual_mode': False  # 自动模式
        }
        
        env = MultiAgentLogisticsEnv(env_config)
        obs, _ = env.reset()
        
        # 快速生成订单
        for step in range(20):
            env._generate_orders()
        
        pending_count = len(env.pending_orders)
        print(f"快速生成后待处理订单: {pending_count}")
        print(f"未超过最大限制: {pending_count <= env.max_pending_orders}")
        
        passed = pending_count <= env.max_pending_orders
        self.test_results.append(("快速订单生成", passed))
        print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
        return passed
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*80)
        print("测试报告汇总")
        print("="*80)
        
        passed_count = sum(1 for _, passed in self.test_results if passed)
        total_count = len(self.test_results)
        
        for name, passed in self.test_results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{name}: {status}")
        
        print("-"*80)
        print(f"总计: {passed_count}/{total_count} 通过 ({passed_count/total_count*100:.1f}%)")
        
        if passed_count == total_count:
            print("\n🎉 所有测试通过！")
        else:
            print(f"\n⚠️ 有 {total_count - passed_count} 个测试失败")


if __name__ == "__main__":
    test_suite = LogisticsTestSuite()
    test_suite.run_all_tests()

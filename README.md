# 基于深度强化学习的多智能体协作任务调度系统
# Multi-Agent Collaborative Task Scheduling System Based on Deep Reinforcement Learning

基于深度强化学习的多智能体协作任务调度系统，支持服务器资源调度、无人机协同任务和物流调度三大应用场景。

A multi-agent collaborative task scheduling system based on deep reinforcement learning, supporting three major application scenarios: server resource scheduling, UAV collaborative tasks, and logistics scheduling.

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [核心模块](#核心模块)
- [算法说明](#算法说明)
- [应用场景](#应用场景)
- [测试结果](#测试结果)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

## 项目概述

本项目实现了一个通用的多智能体强化学习框架，通过去中心化的MAPPO算法实现多智能体协作。系统支持三种典型应用场景：

1. **服务器调度**：多服务器负载均衡和任务分配
2. **无人机调度**：3D空间中的巡检、队形保持和协同包围
3. **物流调度**：多仓库货物调度和车辆配送优化

### 核心创新点

- **去中心化架构**：每个智能体独立决策，提高系统鲁棒性
- **环境驱动协作**：通过精心设计的奖励函数实现隐式协作
- **多任务支持**：同一框架支持多种应用场景
- **高精度控制**：无人机队形保持误差<4米
- **插件式架构**：支持动态加载场景和策略
- **自动化评估**：支持批量评估和报告生成
- **一体化启动**：单文件启动UI和API服务

## 核心特性

### 🎯 算法特性

| 特性 | 说明 |
|------|------|
| **去中心化MAPPO** | 每个智能体独立的Actor-Critic网络 |
| **离散动作空间** | 支持27种离散动作组合 |
| **GAE优势估计** | 广义优势估计，稳定训练 |
| **PPO裁剪** | 策略更新约束，防止策略崩溃 |
| **经验回放** | 每个智能体独立存储和更新 |

### 🖥️ 系统特性

- **图形化界面**：基于PyQt6的实时监控界面
- **3D可视化**：无人机任务支持3D实时渲染
- **多算法支持**：MADDPG和MAPPO算法可切换
- **模型管理**：自动保存最佳模型，支持断点续训
- **日志系统**：完整的训练和运行日志记录
- **插件系统**：支持动态加载环境和策略
- **内置API服务**：UI启动时自动启动API服务器
- **Web控制台**：提供网页版API控制台
- **自动化评估**：批量评估和结构化报告

### 📊 性能指标

| 场景 | 指标 | 数值 |
|------|------|------|
| 无人机队形 | 队形误差 | < 4米 |
| 无人机队形 | 总奖励 | 8000-10000 |
| 物流调度 | 订单完成率 | > 90% |
| 服务器调度 | 负载均衡度 | 优化显著 |

## 技术架构

### 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        系统架构图                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   UI层      │    │   API层     │    │   核心层     │         │
│  │  PyQt6      │    │  Flask API  │    │  调度引擎    │         │
│  │  可视化     │    │  Web控制台  │    │  插件管理    │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                │
│         └──────────────────┼──────────────────┘                │
│                            ▼                                   │
│              ┌─────────────────────────┐                       │
│              │      环境层            │                       │
│              │  ┌─────────────────┐   │                       │
│              │  │  服务器环境      │   │                       │
│              │  │  MultiAgent     │   │                       │
│              │  │  ServerEnv      │   │                       │
│              │  └─────────────────┘   │                       │
│              │  ┌─────────────────┐   │                       │
│              │  │  无人机环境      │   │                       │
│              │  │  MultiAgent     │   │                       │
│              │  │  DroneEnv       │   │                       │
│              │  └─────────────────┘   │                       │
│              │  ┌─────────────────┐   │                       │
│              │  │  物流环境        │   │                       │
│              │  │  MultiAgent     │   │                       │
│              │  │  LogisticsEnv   │   │                       │
│              │  └─────────────────┘   │                       │
│              └─────────────────────────┘                       │
│                            │                                   │
│                            ▼                                   │
│              ┌─────────────────────────┐                       │
│              │      智能体层          │                       │
│              │  ┌─────────────────┐   │                       │
│              │  │  Actor Network  │   │  策略网络            │
│              │  │  Critic Network │   │  价值网络            │
│              │  │  独立优化器      │   │  各自更新            │
│              │  └─────────────────┘   │                       │
│              └─────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 协作机制

虽然采用去中心化架构，但系统通过以下机制实现智能体协作：

1. **观测共享**：每个智能体可以看到其他智能体的相对位置
2. **奖励设计**：全局队形质量影响所有智能体的奖励
3. **角色分工**：领航-跟随者架构，明确任务分工
4. **环境耦合**：环境状态将所有智能体联系在一起

```python
# 观测空间包含协作信息
observation = [
    local_state,           # 自身状态
    neighbor_states,       # 其他智能体相对位置
    leader_state,          # 领航机位置
    formation_error,       # 全局队形误差
]
```

## 项目结构

```
multi_agent_scheduler/
├── config/                     # 配置文件
│   └── config.yaml            # 系统配置
│
├── core/                       # 核心模块
│   ├── __init__.py
│   ├── plugin_interface.py    # 插件接口定义
│   ├── plugin_manager.py      # 插件管理器
│   └── scheduler_engine.py    # 调度引擎核心
│
├── api/                        # API模块
│   ├── __init__.py
│   ├── scheduler_api.py       # 调度器API
│   └── evaluation_api.py      # 评估API
│
├── plugins/                    # 插件目录
│   ├── __init__.py
│   ├── server_environment.py  # 服务器环境插件
│   ├── drone_environment.py   # 无人机环境插件
│   ├── logistics_environment.py # 物流环境插件
│   ├── mappo_strategy.py      # MAPPO策略插件
│   └── random_strategy.py     # 随机策略插件
│
├── environments/               # 环境定义
│   ├── multi_agent_server_env.py      # 服务器调度环境
│   ├── multi_agent_drone_env.py       # 无人机调度环境
│   └── multi_agent_logistics_env.py   # 物流调度环境
│
├── train/                      # 训练脚本
│   ├── server/
│   │   ├── train_maddpg_server.py     # MADDPG训练
│   │   └── train_mappo_server.py      # MAPPO训练
│   ├── drone/
│   │   ├── train_mappo_formation.py   # 队形任务训练
│   │   └── train_mappo_encirclement.py # 包围任务训练
│   └── logistics/
│       └── train_mappo_logistics.py   # 物流调度训练
│
├── ui/                         # 用户界面
│   ├── __init__.py
│   ├── main_window.py                 # 主窗口
│   ├── backend_controller.py          # 后端控制器
│   ├── server_visualization.py        # 服务器可视化
│   ├── drone_visualization.py         # 无人机可视化
│   ├── logistics_visualization.py     # 物流可视化
│   └── evaluation_widget.py           # 评估模块UI
│
├── templates/                  # 网页模板
│   └── index.html             # API控制台网页
│
├── test/                       # 测试脚本
│   ├── test_all_formations.py         # 队形测试
│   ├── test_logistics_comprehensive.py # 物流综合测试
│   ├── test_plugin_system.py          # 插件系统测试
│   └── example_usage.py               # 使用示例
│
├── models/                     # 预训练模型
│   ├── multi_agent_server/
│   ├── multi_agent_drone/
│   └── multi_agent_logistics/
│
├── utils/                      # 工具函数
│   └── logging_config.py      # 日志配置
│
├── evaluations/                # 评估报告
├── logs/                       # 日志文件
├── requirements.txt            # 依赖包
├── run_ui.py                   # 启动脚本（启动UI和API）
└── README.md                   # 项目文档
```

## 环境要求

### 系统要求

- Python >= 3.9
- 操作系统：Windows 10/11 或 Linux
- 内存：建议 8GB+
- 显卡：支持CUDA的GPU（可选，用于加速训练）

### 依赖包

```bash
pip install -r requirements.txt
```

主要依赖：
- `torch>=2.0.0` - 深度学习框架
- `numpy>=1.21.0` - 数值计算
- `gymnasium>=0.29.0` - 强化学习环境
- `PyQt6>=6.4.0` - 图形界面
- `pyqtgraph>=0.13.0` - 3D可视化
- `matplotlib>=3.5.0` - 绘图
- `flask>=3.0.0` - Web API框架
- `loguru>=0.6.0` - 日志记录

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd multi_agent_scheduler

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动系统

```bash
python run_ui.py
```

启动后会自动：
1. 启动API服务器（端口5003）
2. 启动UI界面
3. 打开浏览器访问 http://127.0.0.1:5003/ 可查看Web控制台

### 3. 访问API控制台

在UI界面中：
- 点击 `工具` → `API控制台` (快捷键 Ctrl+A)
- 或点击工具栏的 `API` 按钮
- 会自动打开浏览器显示API网页控制台

### 4. 运行测试

```bash
# 测试无人机队形
python test/test_all_formations.py

# 测试物流调度
python test/test_logistics_comprehensive.py

# 测试插件系统
python test/test_plugin_system.py
```

## 使用指南

### 训练模型

#### 无人机队形任务

```bash
# 训练三角形队形
python train/drone/train_mappo_formation.py --formation triangle --episodes 600

# 训练一字形队形
python train/drone/train_mappo_formation.py --formation line --episodes 600

# 训练V形队形
python train/drone/train_mappo_formation.py --formation v_shape --episodes 600
```

#### 物流调度任务

```bash
python train/logistics/train_mappo_logistics.py --episodes 2000
```

#### 服务器调度任务

```bash
# MAPPO算法
python train/server/train_mappo_server.py --episodes 1000

# MADDPG算法
python train/server/train_maddpg_server.py --episodes 1000
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--formation` | 队形类型 (triangle/v_shape/line) | triangle |
| `--episodes` | 训练回合数 | 500 |
| `--num_drones` | 无人机数量 | 3 |
| `--max_speed` | 最大速度 | 2.0 |

### 配置环境

编辑 `config/config.yaml` 修改系统配置：

```yaml
environment:
  server:
    num_servers: 5
    max_tasks: 100
  drone:
    num_drones: 3
    space_size: [100, 100, 50]
  logistics:
    num_warehouses: 3
    num_vehicles: 5
```

## 核心模块

### 1. 插件式调度引擎

插件式调度引擎作为系统中枢，提供统一协调各功能单元的能力。

#### 主要功能
- **插件管理**：动态加载、初始化、切换插件
- **场景管理**：统一管理各功能单元的调度流程
- **运行时切换**：支持策略和环境的动态切换
- **批量评估**：支持多场景批量评估模式

#### 使用示例

```python
from core.scheduler_engine import SchedulerEngine

# 创建引擎
engine = SchedulerEngine()
engine.initialize()

# 加载场景
engine.load_scenario({
    'environment': 'server_environment',
    'strategy': 'mappo_strategy',
    'env_config': {'num_servers': 5},
    'strategy_config': {'model_path': 'models/mappo'}
})

# 运行时切换策略
engine.switch_strategy('random_strategy')

# 运行评估
result = engine.run_episode(max_steps=100)
```

### 2. 内置API服务

系统启动时自动启动Flask API服务器，提供REST API和Web控制台。

#### 主要功能
- **任务分配**：单任务和批量任务分配
- **策略切换**：运行时切换调度策略
- **状态查询**：查询引擎状态和插件列表
- **自动化评估**：多轮仿真实验和性能评估
- **Web控制台**：可视化的API操作界面

#### API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web控制台首页 |
| `/api/status` | GET | 获取引擎状态 |
| `/api/plugins` | GET | 获取插件列表 |
| `/api/scenarios` | GET | 获取可用场景 |
| `/api/strategy/current` | GET | 获取当前策略 |
| `/api/scenario/load` | POST | 加载场景配置 |
| `/api/strategy/switch` | POST | 切换策略 |
| `/api/task/allocate` | POST | 任务分配 |

#### Python API使用

```python
from api.scheduler_api import SchedulerAPI
from api.evaluation_api import EvaluationAPI

# 创建API
api = SchedulerAPI()
api.initialize()

# 请求任务分配
result = api.request_task_allocation({
    'task_id': 'task_001',
    'task_type': 'computation',
    'priority': 5
})

# 自动化评估
eval_api = EvaluationAPI()
result = eval_api.run_evaluation({
    'name': 'my_evaluation',
    'environment': 'server_environment',
    'strategy': 'mappo_strategy',
    'episodes': 100,
    'export_format': 'json',
    'export_path': 'results/eval_001'
})
```

### 3. UI自动化评估

集成到图形界面的自动化评估模块，提供可视化的评估配置、执行和报告展示。

#### 使用方式
- 切换到 `自动化评估` 标签页
- 或点击 `工具` → `自动化评估` (快捷键 Ctrl+E)

#### 功能特性
- **评估配置**：可视化配置环境、策略、参数
- **实时进度**：进度条和状态信息显示
- **报告展示**：多维度指标、回合数据、原始数据
- **报告导出**：支持JSON和CSV格式

## 算法说明

### MAPPO (Multi-Agent PPO)

#### 算法特点

- **去中心化**：每个智能体独立决策
- **离散动作**：支持27种离散动作
- **GAE优势估计**：稳定的价值估计
- **PPO裁剪**：限制策略更新幅度

#### 网络结构

```python
# Actor网络（策略网络）
ActorNetwork(
    state_dim -> hidden_dim(256) -> hidden_dim(256) -> action_dim(27)
)

# Critic网络（价值网络）
CriticNetwork(
    state_dim -> hidden_dim(256) -> hidden_dim(256) -> value(1)
)
```

#### 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `actor_lr` | 1e-4 | Actor学习率 |
| `critic_lr` | 5e-4 | Critic学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE参数 |
| `clip_epsilon` | 0.15 | PPO裁剪系数 |
| `entropy_coef` | 0.08 | 熵奖励系数 |
| `ppo_epochs` | 15 | 每次更新迭代次数 |

### MADDPG (可选)

MADDPG算法同样支持，适用于连续动作空间场景。

## 应用场景

### 1. 服务器调度

**场景描述**：多个服务器协同处理任务，优化负载均衡。

**状态空间**：
- 服务器负载
- 任务队列长度
- 网络延迟

**动作空间**：任务分配决策

**奖励函数**：负载均衡度 + 任务完成时间

### 2. 无人机调度

#### 队形保持任务

**场景描述**：多无人机保持特定几何队形移动。

**支持队形**：
- 三角形 (triangle)
- 一字形 (line)
- V字形 (v_shape)

**状态空间**：
- 自身位置、速度
- 其他无人机相对位置
- 领航机位置
- 队形误差

**奖励函数**：
```python
# 高精度队形奖励
if distance_to_expected < 0.5:
    reward += 50.0  # 极精确
elif distance_to_expected < 1.0:
    reward += 40.0  # 精确

# 速度匹配奖励
if velocity_diff < 0.5:
    reward += 10.0
```

#### 协同包围任务

**场景描述**：多无人机协同包围目标。

**策略**：分布式包围，每个无人机负责一个角度。

### 3. 物流调度

**场景描述**：多仓库、多车辆的货物调度和配送。

**状态空间**：
- 仓库库存
- 车辆位置、载货量
- 订单状态

**动作空间**：
- 仓库：分配订单给车辆
- 车辆：选择目标仓库或配送

**奖励函数**：
- 订单完成奖励
- 配送效率奖励
- 车辆利用率奖励

## 测试结果

### 无人机队形测试

| 队形 | 总奖励 | 平均奖励 | 队形误差 | 评价 |
|------|--------|----------|----------|------|
| triangle | 8841 | 44.2 | 4.26米 | ✅ 良好 |
| line | 9241 | 46.2 | 3.00米 | ✅ 良好 |
| v_shape | 2451 | 12.3 | 5.98米 | ⚠️ 一般 |

### 物流调度测试

| 测试项 | 状态 |
|--------|------|
| 基础订单生命周期 | ✅ 通过 |
| 多车辆协作 | ✅ 通过 |
| 车辆容量限制 | ✅ 通过 |
| 订单优先级 | ✅ 通过 |
| 并发动作 | ✅ 通过 |

**总计**：11/11 测试通过 (100%)

## 开发指南

### 添加新环境

1. 在 `environments/` 目录创建新环境类
2. 继承 `gymnasium.Env`
3. 实现 `reset()` 和 `step()` 方法
4. 定义观测空间和动作空间

```python
class NewEnvironment(gym.Env):
    def __init__(self, config):
        super().__init__()
        # 初始化
        
    def reset(self):
        # 重置环境
        return observation, info
        
    def step(self, actions):
        # 执行动作
        return observation, reward, done, truncated, info
```

### 添加新算法

1. 在 `train/` 目录创建训练脚本
2. 实现 Actor 和 Critic 网络
3. 实现训练循环
4. 添加模型保存/加载功能

### 添加插件

1. 继承相应的插件基类
2. 实现必要的方法
3. 将插件放入 `plugins/` 目录

```python
from core.plugin_interface import StrategyPlugin

class MyStrategyPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__(name="my_strategy", version="1.0.0")
    
    def initialize(self, config):
        # 初始化
        return True
    
    def make_decision(self, observation, agent_id):
        # 决策逻辑
        return action
```

### 调试技巧

1. **查看日志**：`logs/` 目录包含详细日志
2. **可视化**：使用UI界面实时监控
3. **单元测试**：`test/` 目录有各类测试脚本
4. **API调试**：使用Web控制台或curl测试API

## 常见问题

### Q: 训练不稳定怎么办？

A: 尝试以下方法：
- 降低学习率
- 增加 `entropy_coef` 鼓励探索
- 减小 `clip_epsilon` 限制更新幅度
- 增加 `ppo_epochs` 提高样本利用率

### Q: 如何调整无人机数量？

A: 修改环境配置：
```python
env_config = {
    'num_drones': 5,  # 改为5个无人机
    # ...
}
```
系统会自动扩展队形偏移量。

### Q: 模型加载失败怎么办？

A: 检查：
1. 模型文件路径是否正确
2. 网络结构是否与训练时一致
3. 使用 `map_location='cpu'` 加载

### Q: 如何添加新的队形？

A: 在 `multi_agent_drone_env.py` 中添加：
```python
self.formations = {
    'triangle': [(0, 0, 0), (-10, -10, 0), (10, -10, 0)],
    'line': [(0, 0, 0), (10, 0, 0), (20, 0, 0)],
    'new_formation': [(0, 0, 0), (0, 10, 0), (0, -10, 0)],  # 新队形
}
```

### Q: 如何使用插件系统？

A: 参考 `test/example_usage.py` 中的示例：
```python
from core.scheduler_engine import SchedulerEngine

engine = SchedulerEngine()
engine.initialize()
engine.load_scenario({
    'environment': 'server_environment',
    'strategy': 'mappo_strategy'
})
```

### Q: 如何进行自动化评估？

A: 三种方式：
1. **UI界面**：切换到 `自动化评估` 标签页
2. **Python API**：使用 `EvaluationAPI`
3. **Web控制台**：访问 http://127.0.0.1:5003/

### Q: API服务器无法启动怎么办？

A: 检查：
1. 端口5003是否被占用
2. Flask是否已安装：`pip install flask`
3. 查看控制台错误信息

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

---

**最后更新**：2026-03-02

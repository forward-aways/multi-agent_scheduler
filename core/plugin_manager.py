"""
插件管理器模块
负责插件的加载、注册、管理和运行时切换
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
import logging

from .plugin_interface import PluginInterface, StrategyPlugin, EnvironmentPlugin, EvaluationPlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """插件管理器"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        """
        初始化插件管理器
        
        Args:
            plugin_dirs: 插件目录列表
        """
        self.plugins: Dict[str, PluginInterface] = {}
        self.active_plugins: Dict[str, str] = {}  # category -> plugin_name
        self.plugin_classes: Dict[str, Type[PluginInterface]] = {}
        
        # 默认插件目录
        if plugin_dirs is None:
            plugin_dirs = ['plugins']
        
        self.plugin_dirs = plugin_dirs
        
        # 确保插件目录在Python路径中
        for plugin_dir in plugin_dirs:
            if os.path.exists(plugin_dir) and plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
    
    def discover_plugins(self) -> List[str]:
        """
        发现可用插件
        
        Returns:
            发现的插件名称列表
        """
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
            
            # 扫描插件目录
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                # 检查是否是Python文件
                if os.path.isfile(item_path) and item.endswith('.py') and not item.startswith('__'):
                    plugin_name = item[:-3]  # 去掉.py后缀
                    discovered.append(plugin_name)
                
                # 检查是否是包
                elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
                    discovered.append(item)
        
        logger.info(f"发现 {len(discovered)} 个插件: {discovered}")
        return discovered
    
    def load_plugin(self, plugin_name: str, plugin_dir: str = None) -> bool:
        """
        加载插件
        
        Args:
            plugin_name: 插件名称
            plugin_dir: 插件所在目录（可选）
            
        Returns:
            加载是否成功
        """
        try:
            # 如果已加载，先卸载
            if plugin_name in self.plugins:
                self.unload_plugin(plugin_name)
            
            # 导入插件模块
            module = importlib.import_module(plugin_name)
            
            # 查找插件类
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface and
                    obj != StrategyPlugin and
                    obj != EnvironmentPlugin and
                    obj != EvaluationPlugin):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                logger.error(f"在插件 {plugin_name} 中未找到插件类")
                return False
            
            # 实例化插件
            plugin = plugin_class()
            self.plugins[plugin_name] = plugin
            self.plugin_classes[plugin_name] = plugin_class
            
            logger.info(f"成功加载插件: {plugin_name} (版本: {plugin.version})")
            return True
            
        except Exception as e:
            logger.error(f"加载插件 {plugin_name} 失败: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        卸载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            卸载是否成功
        """
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            
            # 如果插件处于激活状态，先关闭
            if plugin.is_active:
                plugin.shutdown()
            
            # 从激活列表中移除
            for category, name in list(self.active_plugins.items()):
                if name == plugin_name:
                    del self.active_plugins[category]
            
            # 移除插件
            del self.plugins[plugin_name]
            if plugin_name in self.plugin_classes:
                del self.plugin_classes[plugin_name]
            
            logger.info(f"成功卸载插件: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"卸载插件 {plugin_name} 失败: {e}")
            return False
    
    def initialize_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """
        初始化插件
        
        Args:
            plugin_name: 插件名称
            config: 配置参数
            
        Returns:
            初始化是否成功
        """
        if plugin_name not in self.plugins:
            logger.error(f"插件 {plugin_name} 未加载")
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            success = plugin.initialize(config or {})
            
            if success:
                plugin.is_active = True
                logger.info(f"成功初始化插件: {plugin_name}")
            else:
                logger.error(f"初始化插件 {plugin_name} 失败")
            
            return success
            
        except Exception as e:
            logger.error(f"初始化插件 {plugin_name} 时出错: {e}")
            return False
    
    def activate_plugin(self, plugin_name: str, category: str = None) -> bool:
        """
        激活插件（设置为当前使用的插件）
        
        Args:
            plugin_name: 插件名称
            category: 插件类别（如 'strategy', 'environment'）
            
        Returns:
            激活是否成功
        """
        if plugin_name not in self.plugins:
            logger.error(f"插件 {plugin_name} 未加载")
            return False
        
        plugin = self.plugins[plugin_name]
        
        if not plugin.is_active:
            logger.error(f"插件 {plugin_name} 未初始化")
            return False
        
        # 确定类别
        if category is None:
            if isinstance(plugin, StrategyPlugin):
                category = 'strategy'
            elif isinstance(plugin, EnvironmentPlugin):
                category = 'environment'
            elif isinstance(plugin, EvaluationPlugin):
                category = 'evaluation'
            else:
                category = 'general'
        
        # 设置为激活状态
        self.active_plugins[category] = plugin_name
        logger.info(f"激活插件: {plugin_name} (类别: {category})")
        
        return True
    
    def get_active_plugin(self, category: str) -> Optional[PluginInterface]:
        """
        获取当前激活的插件
        
        Args:
            category: 插件类别
            
        Returns:
            激活的插件实例，如果没有则返回None
        """
        if category not in self.active_plugins:
            return None
        
        plugin_name = self.active_plugins[category]
        return self.plugins.get(plugin_name)
    
    def execute_plugin(self, plugin_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定插件
        
        Args:
            plugin_name: 插件名称
            context: 执行上下文
            
        Returns:
            执行结果
        """
        if plugin_name not in self.plugins:
            return {'success': False, 'error': f'Plugin {plugin_name} not found'}
        
        plugin = self.plugins[plugin_name]
        
        if not plugin.is_active:
            return {'success': False, 'error': f'Plugin {plugin_name} is not active'}
        
        try:
            return plugin.execute(context)
        except Exception as e:
            logger.error(f"执行插件 {plugin_name} 时出错: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_plugins(self, category: str = None) -> List[Dict[str, Any]]:
        """
        列出所有插件
        
        Args:
            category: 插件类别过滤器
            
        Returns:
            插件信息列表
        """
        plugins_info = []
        
        for name, plugin in self.plugins.items():
            info = plugin.get_info()
            
            # 确定类别
            if isinstance(plugin, StrategyPlugin):
                info['category'] = 'strategy'
            elif isinstance(plugin, EnvironmentPlugin):
                info['category'] = 'environment'
            elif isinstance(plugin, EvaluationPlugin):
                info['category'] = 'evaluation'
            else:
                info['category'] = 'general'
            
            # 检查是否是激活状态
            info['is_active_plugin'] = name in self.active_plugins.get(info['category'], '')
            
            # 类别过滤
            if category is None or info['category'] == category:
                plugins_info.append(info)
        
        return plugins_info
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        获取插件详细信息
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件信息，如果不存在则返回None
        """
        if plugin_name not in self.plugins:
            return None
        
        return self.plugins[plugin_name].get_info()
    
    def switch_strategy(self, strategy_name: str) -> bool:
        """
        切换策略（便捷方法）
        
        Args:
            strategy_name: 策略插件名称
            
        Returns:
            切换是否成功
        """
        return self.activate_plugin(strategy_name, 'strategy')
    
    def switch_environment(self, env_name: str) -> bool:
        """
        切换环境（便捷方法）
        
        Args:
            env_name: 环境插件名称
            
        Returns:
            切换是否成功
        """
        return self.activate_plugin(env_name, 'environment')
    
    def shutdown_all(self):
        """关闭所有插件"""
        for name, plugin in list(self.plugins.items()):
            try:
                if plugin.is_active:
                    plugin.shutdown()
                    plugin.is_active = False
                logger.info(f"关闭插件: {name}")
            except Exception as e:
                logger.error(f"关闭插件 {name} 时出错: {e}")
        
        self.plugins.clear()
        self.active_plugins.clear()
        logger.info("所有插件已关闭")

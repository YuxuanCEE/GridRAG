# -*- coding: utf-8 -*-
"""
优化模型基类
定义所有优化模型的通用接口和方法
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseOptimizationModel(ABC):
    """优化模型基类"""
    
    def __init__(self, name: str, config: dict):
        """
        初始化模型
        
        Args:
            name: 模型名称
            config: 配置字典
        """
        self.name = name
        self.config = config
        self.model = None
        self.solution = None
        self.solve_time = 0.0
        self.status = "not_solved"
        
        # 结果统计
        self.statistics = {
            "build_time": 0.0,
            "solve_time": 0.0,
            "total_time": 0.0,
            "n_variables": 0,
            "n_constraints": 0,
            "n_binary_vars": 0,
            "objective_value": None,
            "solver_status": None,
        }
    
    @abstractmethod
    def build_model(self, **kwargs):
        """
        构建优化模型
        
        Returns:
            构建好的模型对象
        """
        pass
    
    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        求解优化模型
        
        Returns:
            求解结果字典
        """
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        获取优化结果
        
        Returns:
            结果字典，包含决策变量值、目标函数值等
        """
        pass
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        完整运行：构建模型 + 求解 + 获取结果
        
        Returns:
            包含结果和统计信息的字典
        """
        total_start = time.time()
        
        # 构建模型
        build_start = time.time()
        self.build_model(**kwargs)
        self.statistics["build_time"] = time.time() - build_start
        
        # 求解模型
        solve_start = time.time()
        self.solve(**kwargs)
        self.statistics["solve_time"] = time.time() - solve_start
        
        # 获取结果
        results = self.get_results()
        
        self.statistics["total_time"] = time.time() - total_start
        
        return {
            "results": results,
            "statistics": self.statistics,
        }
    
    def save_results(self, output_dir: Path, prefix: str = ""):
        """
        保存优化结果到文件
        
        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        import json
        import numpy as np
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = self.get_results()
        
        # 转换numpy数组为列表以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        serializable_stats = convert_to_serializable(self.statistics)
        
        # 保存结果
        filename = f"{prefix}_{self.name}_results.json" if prefix else f"{self.name}_results.json"
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": self.name,
                "results": serializable_results,
                "statistics": serializable_stats,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_dir / filename}")
    
    def print_statistics(self):
        """打印求解统计信息"""
        print("\n" + "=" * 50)
        print(f"模型: {self.name}")
        print("=" * 50)
        print(f"模型构建时间: {self.statistics['build_time']:.3f} 秒")
        print(f"求解时间: {self.statistics['solve_time']:.3f} 秒")
        print(f"总运行时间: {self.statistics['total_time']:.3f} 秒")
        print(f"变量数: {self.statistics['n_variables']}")
        print(f"约束数: {self.statistics['n_constraints']}")
        print(f"二元变量数: {self.statistics['n_binary_vars']}")
        print(f"目标函数值: {self.statistics['objective_value']}")
        print(f"求解状态: {self.statistics['solver_status']}")
        print("=" * 50)

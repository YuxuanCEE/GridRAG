# -*- coding: utf-8 -*-
"""
求解器封装模块
支持多种优化求解器的统一接口
"""

import time
from typing import Dict, Any, Optional
import pyomo.environ as pyo


class OptimizationSolver:
    """优化求解器封装类"""
    
    SUPPORTED_SOLVERS = ["gurobi", "cplex", "glpk", "cbc", "ipopt"]
    
    def __init__(self, solver_name: str = "gurobi", **options):
        """
        初始化求解器
        
        Args:
            solver_name: 求解器名称
            **options: 求解器选项
        """
        self.solver_name = solver_name.lower()
        self.options = options
        self.solver = None
        self._create_solver()
    
    def _create_solver(self):
        """创建求解器实例"""
        if self.solver_name not in self.SUPPORTED_SOLVERS:
            raise ValueError(f"不支持的求解器: {self.solver_name}")
        
        self.solver = pyo.SolverFactory(self.solver_name)
        
        if self.solver is None or not self.solver.available():
            raise RuntimeError(f"求解器 {self.solver_name} 不可用，请检查安装")
        
        # 设置默认选项
        self._set_default_options()
    
    def _set_default_options(self):
        """设置默认求解器选项"""
        if self.solver_name == "gurobi":
            # Gurobi默认选项
            default_opts = {
                "TimeLimit": self.options.get("time_limit", 300),
                "MIPGap": self.options.get("mip_gap", 1e-4),
                "OutputFlag": 1 if self.options.get("verbose", True) else 0,
                "Threads": self.options.get("threads", 0),  # 0表示自动
            }
            for key, value in default_opts.items():
                self.solver.options[key] = value
        
        elif self.solver_name == "cplex":
            default_opts = {
                "timelimit": self.options.get("time_limit", 300),
                "mipgap": self.options.get("mip_gap", 1e-4),
            }
            for key, value in default_opts.items():
                self.solver.options[key] = value
        
        elif self.solver_name == "glpk":
            default_opts = {
                "tmlim": self.options.get("time_limit", 300),
                "mipgap": self.options.get("mip_gap", 1e-4),
            }
            for key, value in default_opts.items():
                self.solver.options[key] = value
    
    def solve(self, model: pyo.ConcreteModel, 
              tee: bool = True) -> Dict[str, Any]:
        """
        求解优化模型
        
        Args:
            model: Pyomo模型
            tee: 是否显示求解过程
        
        Returns:
            求解结果字典
        """
        print(f"\n{'='*50}")
        print(f"使用 {self.solver_name.upper()} 求解器求解")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            results = self.solver.solve(model, tee=tee)
            solve_time = time.time() - start_time
            
            # 解析结果
            status = str(results.solver.status)
            termination = str(results.solver.termination_condition)
            
            is_optimal = (results.solver.termination_condition == 
                         pyo.TerminationCondition.optimal)
            
            objective_value = None
            if is_optimal or results.solver.termination_condition == pyo.TerminationCondition.feasible:
                try:
                    objective_value = pyo.value(model.obj)
                except:
                    pass
            
            result_dict = {
                "success": is_optimal,
                "status": status,
                "termination_condition": termination,
                "solve_time": solve_time,
                "objective_value": objective_value,
                "solver_name": self.solver_name,
            }
            
            # 打印摘要
            print(f"\n求解完成!")
            print(f"  状态: {status}")
            print(f"  终止条件: {termination}")
            print(f"  求解时间: {solve_time:.3f} 秒")
            if objective_value is not None:
                print(f"  目标值: {objective_value:.6f}")
            
            return result_dict
            
        except Exception as e:
            solve_time = time.time() - start_time
            print(f"\n求解出错: {str(e)}")
            return {
                "success": False,
                "status": "error",
                "termination_condition": str(e),
                "solve_time": solve_time,
                "objective_value": None,
                "solver_name": self.solver_name,
            }
    
    def set_option(self, key: str, value: Any):
        """设置求解器选项"""
        self.solver.options[key] = value
    
    @staticmethod
    def check_solver_availability(solver_name: str) -> bool:
        """检查求解器是否可用"""
        try:
            solver = pyo.SolverFactory(solver_name)
            return solver is not None and solver.available()
        except:
            return False
    
    @staticmethod
    def list_available_solvers() -> list:
        """列出所有可用的求解器"""
        available = []
        for solver in OptimizationSolver.SUPPORTED_SOLVERS:
            if OptimizationSolver.check_solver_availability(solver):
                available.append(solver)
        return available


def get_solver(solver_name: str = "gurobi", **options) -> OptimizationSolver:
    """获取求解器实例"""
    return OptimizationSolver(solver_name, **options)


if __name__ == "__main__":
    # 测试求解器可用性
    print("检查求解器可用性:")
    print("-" * 30)
    
    for solver in OptimizationSolver.SUPPORTED_SOLVERS:
        available = OptimizationSolver.check_solver_availability(solver)
        status = "✓ 可用" if available else "✗ 不可用"
        print(f"  {solver}: {status}")
    
    print("\n可用求解器列表:", OptimizationSolver.list_available_solvers())

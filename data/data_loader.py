# -*- coding: utf-8 -*-
"""
数据加载模块
负责加载PV、WT功率曲线和生成负荷曲线

支持的数据格式:
- 单个CSV文件包含所有DG节点数据
- 列名格式: node_XX_PV, node_XX_wind 等
- 数据可以是标幺值(0-1)或实际功率(MW)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, PROFILES_DIR


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or get_config()
        self.profiles_dir = Path(self.config["paths"]["profiles"])
        
        # 数据缓存
        self._dg_data_cache = None
        self._load_curve = None
    
    def load_dg_data_from_file(self, filename: str, 
                                scenario_id: Optional[int] = None) -> pd.DataFrame:
        """
        从CSV文件加载分布式电源数据
        
        Args:
            filename: CSV文件名
            scenario_id: 场景ID，如果文件包含多个场景
        
        Returns:
            DataFrame包含所有DG数据
        """
        file_path = self.profiles_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 读取CSV
        df = pd.read_csv(file_path)
        
        # 处理BOM标记（如果有）
        if df.columns[0].startswith('\ufeff'):
            df.columns = [col.replace('\ufeff', '') for col in df.columns]
        
        # 如果有scenario_id列且指定了场景
        if 'scenario_id' in df.columns and scenario_id is not None:
            df = df[df['scenario_id'] == scenario_id].copy()
        
        # 解析时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def load_pv_data(self, filename: Optional[str] = None,
                     scenario_id: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        加载光伏功率数据
        
        Args:
            filename: 数据文件名，如果为None则使用配置中的默认文件
            scenario_id: 场景ID
        
        Returns:
            字典 {节点编号(0-indexed): 功率数组(MW)}
        """
        if filename is None:
            filename = self.config["data"]["dg_data_file"]
        
        pv_config = self.config["devices"]["pv"]
        buses = pv_config["buses"]
        columns = pv_config["columns"]
        capacities = pv_config["capacity"]
        data_type = self.config["data"].get("data_type", "pu")
        
        # 加载数据
        df = self.load_dg_data_from_file(filename, scenario_id)
        
        pv_data = {}
        for bus, col, capacity in zip(buses, columns, capacities):
            if col in df.columns:
                power = df[col].values
                
                # 如果是标幺值，乘以容量转换为MW
                if data_type == "pu":
                    power = power * capacity
                
                pv_data[bus] = power
            else:
                print(f"警告: PV列 '{col}' 不存在，使用模拟数据")
                pv_data[bus] = self._generate_synthetic_pv(capacity, n_periods=len(df))
        
        return pv_data
    
    def load_wt_data(self, filename: Optional[str] = None,
                     scenario_id: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        加载风电功率数据
        
        Args:
            filename: 数据文件名
            scenario_id: 场景ID
        
        Returns:
            字典 {节点编号(0-indexed): 功率数组(MW)}
        """
        if filename is None:
            filename = self.config["data"]["dg_data_file"]
        
        wt_config = self.config["devices"]["wt"]
        buses = wt_config["buses"]
        columns = wt_config["columns"]
        capacities = wt_config["capacity"]
        data_type = self.config["data"].get("data_type", "pu")
        
        # 加载数据
        df = self.load_dg_data_from_file(filename, scenario_id)
        
        wt_data = {}
        for bus, col, capacity in zip(buses, columns, capacities):
            if col in df.columns:
                power = df[col].values
                
                # 如果是标幺值，乘以容量转换为MW
                if data_type == "pu":
                    power = power * capacity
                
                wt_data[bus] = power
            else:
                print(f"警告: WT列 '{col}' 不存在，使用模拟数据")
                wt_data[bus] = self._generate_synthetic_wt(capacity, n_periods=len(df))
        
        return wt_data
    
    def get_load_curve(self, n_periods: int = 96) -> np.ndarray:
        """
        获取负荷变化曲线（标准化因子）
        
        Args:
            n_periods: 时段数（96对应15分钟分辨率的一天）
        
        Returns:
            负荷因子数组 (n_periods,)，范围约0.3-1.0
        """
        if self._load_curve is not None and len(self._load_curve) == n_periods:
            return self._load_curve
        
        # 生成典型日负荷曲线（归一化）
        # 基于论文附录A3的负荷变化率曲线
        hours = np.linspace(0, 24, n_periods, endpoint=False)
        
        # 典型居民/商业混合负荷曲线
        load_curve = (
            0.4  # 基础负荷
            + 0.15 * np.sin(np.pi * (hours - 6) / 12) * (hours >= 6) * (hours <= 22)  # 白天高峰
            + 0.25 * np.exp(-((hours - 12) ** 2) / 8)  # 中午高峰
            + 0.2 * np.exp(-((hours - 19) ** 2) / 4)   # 晚高峰
            - 0.1 * np.exp(-((hours - 4) ** 2) / 4)    # 凌晨低谷
        )
        
        # 归一化到0.3-1.0范围
        load_curve = 0.3 + 0.7 * (load_curve - load_curve.min()) / (load_curve.max() - load_curve.min())
        
        self._load_curve = load_curve
        return load_curve
    
    def _generate_synthetic_pv(self, capacity: float, n_periods: int = 96) -> np.ndarray:
        """
        生成模拟光伏功率曲线
        
        Args:
            capacity: 装机容量 MW
            n_periods: 时段数
        
        Returns:
            功率数组 (MW)
        """
        hours = np.linspace(0, 24, n_periods, endpoint=False)
        
        # 光伏出力曲线（日出-日落的钟形曲线）
        sunrise, sunset = 6, 18
        pv_curve = np.zeros(n_periods)
        
        daylight_mask = (hours >= sunrise) & (hours <= sunset)
        daylight_hours = hours[daylight_mask]
        
        # 使用正弦函数模拟日出到日落
        pv_output = np.sin(np.pi * (daylight_hours - sunrise) / (sunset - sunrise))
        pv_output = np.maximum(pv_output, 0)
        
        # 添加随机波动（云层遮挡等）
        np.random.seed(42)
        noise = 1 + 0.1 * np.random.randn(len(pv_output))
        pv_output = pv_output * np.clip(noise, 0.7, 1.1)
        
        pv_curve[daylight_mask] = pv_output * capacity * 0.8  # 0.8为容量因子
        
        return pv_curve
    
    def _generate_synthetic_wt(self, capacity: float, n_periods: int = 96) -> np.ndarray:
        """
        生成模拟风电功率曲线
        
        Args:
            capacity: 装机容量 MW
            n_periods: 时段数
        
        Returns:
            功率数组 (MW)
        """
        np.random.seed(123)
        
        # 风电出力更随机，但有一定的自相关性
        # 使用AR(1)过程生成
        wt_curve = np.zeros(n_periods)
        wt_curve[0] = 0.3 + 0.4 * np.random.rand()
        
        for t in range(1, n_periods):
            # AR(1)系数
            phi = 0.9
            innovation = 0.15 * np.random.randn()
            wt_curve[t] = phi * wt_curve[t-1] + (1 - phi) * 0.4 + innovation
        
        # 限制在合理范围
        wt_curve = np.clip(wt_curve, 0.05, 0.95) * capacity
        
        return wt_curve
    
    def get_scenario_data(self, filename: Optional[str] = None,
                          scenario_id: Optional[int] = None,
                          n_periods: int = 96) -> Dict:
        """
        获取完整场景数据
        
        Args:
            filename: 数据文件名，如果为None则使用配置中的默认文件
            scenario_id: 场景ID
            n_periods: 期望的时段数
        
        Returns:
            包含PV、WT、负荷数据的字典
        """
        if filename is None:
            filename = self.config["data"]["dg_data_file"]
        
        pv_data = self.load_pv_data(filename, scenario_id)
        wt_data = self.load_wt_data(filename, scenario_id)
        load_curve = self.get_load_curve(n_periods)
        
        # 验证数据长度并进行插值（如果需要）
        for bus, data in pv_data.items():
            if len(data) != n_periods:
                print(f"警告: PV节点{bus}数据长度{len(data)}与期望{n_periods}不符，进行插值")
                pv_data[bus] = np.interp(
                    np.linspace(0, 1, n_periods),
                    np.linspace(0, 1, len(data)),
                    data
                )
        
        for bus, data in wt_data.items():
            if len(data) != n_periods:
                print(f"警告: WT节点{bus}数据长度{len(data)}与期望{n_periods}不符，进行插值")
                wt_data[bus] = np.interp(
                    np.linspace(0, 1, n_periods),
                    np.linspace(0, 1, len(data)),
                    data
                )
        
        # 获取日期信息
        try:
            df = self.load_dg_data_from_file(filename, scenario_id)
            if 'timestamp' in df.columns:
                date_str = df['timestamp'].iloc[0].strftime("%Y-%m-%d")
            else:
                date_str = "unknown"
        except:
            date_str = "unknown"
        
        return {
            "filename": filename,
            "scenario_id": scenario_id,
            "date": date_str,
            "n_periods": n_periods,
            "pv": pv_data,
            "wt": wt_data,
            "load_factor": load_curve,
        }
    
    def list_available_scenarios(self, filename: Optional[str] = None) -> List[int]:
        """
        列出文件中可用的场景ID
        
        Args:
            filename: 数据文件名
        
        Returns:
            场景ID列表
        """
        if filename is None:
            filename = self.config["data"]["dg_data_file"]
        
        try:
            df = self.load_dg_data_from_file(filename)
            if 'scenario_id' in df.columns:
                return sorted(df['scenario_id'].unique().tolist())
            else:
                return [0]  # 没有scenario_id列，返回默认值
        except:
            return []
    
    def print_data_summary(self, filename: Optional[str] = None,
                           scenario_id: Optional[int] = None):
        """
        打印数据摘要信息
        """
        if filename is None:
            filename = self.config["data"]["dg_data_file"]
        
        print("\n" + "=" * 50)
        print("数据摘要")
        print("=" * 50)
        print(f"数据文件: {filename}")
        
        try:
            df = self.load_dg_data_from_file(filename, scenario_id)
            print(f"总行数: {len(df)}")
            print(f"列名: {list(df.columns)}")
            
            if 'timestamp' in df.columns:
                print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            
            # 打印各列统计
            print("\n各列数据统计:")
            for col in df.columns:
                if col not in ['scenario_id', 'timestamp']:
                    print(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
        
        print("=" * 50)


def get_data_loader(config: Optional[dict] = None) -> DataLoader:
    """获取数据加载器实例"""
    return DataLoader(config)


if __name__ == "__main__":
    # 测试数据加载
    loader = get_data_loader()
    
    # 打印数据摘要
    loader.print_data_summary()
    
    # 测试获取场景数据
    try:
        scenario = loader.get_scenario_data(n_periods=96)
        
        print("\n场景数据摘要:")
        print(f"  日期: {scenario['date']}")
        print(f"  时段数: {scenario['n_periods']}")
        print(f"  PV节点: {list(scenario['pv'].keys())}")
        print(f"  WT节点: {list(scenario['wt'].keys())}")
        
        # 打印部分数据
        for bus, data in scenario['pv'].items():
            print(f"  PV节点{bus}: 最大{data.max():.4f}MW, 均值{data.mean():.4f}MW")
        
        for bus, data in scenario['wt'].items():
            print(f"  WT节点{bus}: 最大{data.max():.4f}MW, 均值{data.mean():.4f}MW")
        
        print(f"  负荷因子: 最小{scenario['load_factor'].min():.3f}, 最大{scenario['load_factor'].max():.3f}")
        
    except FileNotFoundError as e:
        print(f"\n文件未找到: {e}")
        print("请将数据文件放入 data/profiles/ 目录")

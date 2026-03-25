# -*- coding: utf-8 -*-
"""utils.result_loader

统一的结果文件加载工具。

默认结果路径规则（与 config_networks.get_result_filename 一致）：
    opt_results/{task}/{network}_{task}_scenario_{scenario_id}_results.json

用法:
    loader = ResultLoader(project_root=Path(__file__).resolve().parents[1])
    data = loader.load(network="ieee33", task="vvc", scenario_id="004")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from config_networks import get_result_filename


@dataclass
class ResultLoader:
    """加载离线保存的优化结果JSON。"""

    project_root: Optional[Path] = None
    results_root: Optional[Path] = None  # 默认: {project_root}/opt_results

    def __post_init__(self):
        if self.project_root is None:
            # utils/ 位于项目根目录下
            self.project_root = Path(__file__).resolve().parents[1]
        if self.results_root is None:
            self.results_root = Path(self.project_root) / "opt_results"

    def build_path(self, network: str, task: str, scenario_id: str) -> Path:
        network = network.lower()
        task = task.lower()
        filename = get_result_filename(network, task, scenario_id, suffix="results")
        return Path(self.results_root) / task / filename

    def load(self, network: str, task: str, scenario_id: str) -> Dict[str, Any]:
        """加载历史结果JSON为dict。"""
        path = self.build_path(network, task, scenario_id)
        if not path.exists():
            raise FileNotFoundError(f"历史结果文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

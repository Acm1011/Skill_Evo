#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Configuration Module

This module provides centralized GPU configuration for the Self-evolving-Agent project.
GPU assignments can be controlled via environment variables, allowing dynamic allocation
without modifying the source code.

Environment Variables:
    - SE_N_GPUS: Total number of GPUs to use (4 or 8)
    - SE_GPU_IDS: Comma-separated GPU IDs (e.g., "0,1,2,3")
    - SE_CHALLENGER_GPUS: GPUs for Challenger training
    - SE_REWARD_GPUS: GPUs for Reward Server
    - SE_SOLVER_GPUS: GPUs for Solver training
    - SE_REWARD_PORTS: Comma-separated ports for Reward Server (e.g., "5000,5001")
    - SE_REWARD_BASE_PORT: Base port for Reward Server (default: 5000)
"""

import os
from typing import List, Tuple, Optional


def get_env_list(var_name: str, default: List[str] = None) -> List[str]:
    """Get a list from environment variable (comma-separated)."""
    value = os.environ.get(var_name, "")
    if not value:
        return default if default is not None else []
    return [x.strip() for x in value.split(",") if x.strip()]


def get_env_int(var_name: str, default: int = 0) -> int:
    """Get an integer from environment variable."""
    try:
        return int(os.environ.get(var_name, str(default)))
    except (ValueError, TypeError):
        return default


class GPUConfig:
    """GPU Configuration class for the Self-evolving-Agent project."""
    
    def __init__(self):
        self.reload()
    
    def reload(self):
        """Reload configuration from environment variables."""
        # Total GPUs
        self.n_gpus = get_env_int("SE_N_GPUS", 4)
        
        # GPU IDs (if specified, use them; otherwise auto-generate)
        gpu_ids_str = get_env_list("SE_GPU_IDS")
        if gpu_ids_str:
            self.gpu_ids = [int(x) for x in gpu_ids_str]
        else:
            self.gpu_ids = list(range(self.n_gpus))
        
        # Reward server ports
        self.reward_base_port = get_env_int("SE_REWARD_BASE_PORT", 5000)
        reward_ports_str = get_env_list("SE_REWARD_PORTS")
        if reward_ports_str:
            self.reward_ports = [int(x) for x in reward_ports_str]
        else:
            # Default: 2 ports for n_gpus=4, 4 ports for n_gpus=8
            n_reward_servers = max(2, self.n_gpus // 2)
            self.reward_ports = [self.reward_base_port + i for i in range(n_reward_servers)]
        
        # Calculate GPU splits
        self._calculate_gpu_splits()
    
    def _calculate_gpu_splits(self):
        """Calculate GPU splits for challenger, reward, and solver."""
        n = self.n_gpus
        half = n // 2
        
        # Default split: first half for challenger/gen_query, second half for reward
        # For solver: use all GPUs
        
        # Challenger GPUs (first half)
        challenger_gpus_str = get_env_list("SE_CHALLENGER_GPUS")
        if challenger_gpus_str:
            self.challenger_gpus = [int(x) for x in challenger_gpus_str]
        else:
            self.challenger_gpus = self.gpu_ids[:half]
        
        # Reward GPUs (second half)
        reward_gpus_str = get_env_list("SE_REWARD_GPUS")
        if reward_gpus_str:
            self.reward_gpus = [int(x) for x in reward_gpus_str]
        else:
            self.reward_gpus = self.gpu_ids[half:]
        
        # Solver GPUs (all)
        solver_gpus_str = get_env_list("SE_SOLVER_GPUS")
        if solver_gpus_str:
            self.solver_gpus = [int(x) for x in solver_gpus_str]
        else:
            self.solver_gpus = self.gpu_ids[:]
        
        # Gen query GPUs (all, for parallel generation)
        gen_query_gpus_str = get_env_list("SE_GEN_QUERY_GPUS")
        if gen_query_gpus_str:
            self.gen_query_gpus = [int(x) for x in gen_query_gpus_str]
        else:
            self.gen_query_gpus = self.gpu_ids[:]
    
    @property
    def challenger_gpus_str(self) -> str:
        """Get challenger GPUs as comma-separated string."""
        return ",".join(str(g) for g in self.challenger_gpus)
    
    @property
    def reward_gpus_str(self) -> str:
        """Get reward GPUs as comma-separated string."""
        return ",".join(str(g) for g in self.reward_gpus)
    
    @property
    def solver_gpus_str(self) -> str:
        """Get solver GPUs as comma-separated string."""
        return ",".join(str(g) for g in self.solver_gpus)
    
    @property
    def gen_query_gpus_str(self) -> str:
        """Get gen_query GPUs as comma-separated string."""
        return ",".join(str(g) for g in self.gen_query_gpus)
    
    @property
    def n_challenger_gpus(self) -> int:
        """Get number of challenger GPUs."""
        return len(self.challenger_gpus)
    
    @property
    def n_reward_gpus(self) -> int:
        """Get number of reward GPUs."""
        return len(self.reward_gpus)
    
    @property
    def n_solver_gpus(self) -> int:
        """Get number of solver GPUs."""
        return len(self.solver_gpus)
    
    @property
    def n_reward_servers(self) -> int:
        """Get number of reward servers."""
        return len(self.reward_ports)
    
    def get_reward_gpu_port_pairs(self) -> List[Tuple[int, int]]:
        """Get list of (gpu_id, port) pairs for reward servers."""
        pairs = []
        for i, port in enumerate(self.reward_ports):
            gpu_idx = i % len(self.reward_gpus)
            pairs.append((self.reward_gpus[gpu_idx], port))
        return pairs
    
    def __repr__(self) -> str:
        return (
            f"GPUConfig(\n"
            f"  n_gpus={self.n_gpus},\n"
            f"  gpu_ids={self.gpu_ids},\n"
            f"  challenger_gpus={self.challenger_gpus},\n"
            f"  reward_gpus={self.reward_gpus},\n"
            f"  solver_gpus={self.solver_gpus},\n"
            f"  gen_query_gpus={self.gen_query_gpus},\n"
            f"  reward_ports={self.reward_ports}\n"
            f")"
        )


# Global instance
_config: Optional[GPUConfig] = None


def get_gpu_config() -> GPUConfig:
    """Get the global GPU configuration instance."""
    global _config
    if _config is None:
        _config = GPUConfig()
    return _config


def reload_gpu_config() -> GPUConfig:
    """Reload the GPU configuration from environment variables."""
    global _config
    _config = GPUConfig()
    return _config


# Export functions for shell scripts
def print_gpu_config_for_shell():
    """Print GPU configuration in a format suitable for shell scripts."""
    config = get_gpu_config()
    print(f"export SE_N_GPUS={config.n_gpus}")
    print(f"export SE_GPU_IDS={','.join(str(g) for g in config.gpu_ids)}")
    print(f"export SE_CHALLENGER_GPUS={config.challenger_gpus_str}")
    print(f"export SE_REWARD_GPUS={config.reward_gpus_str}")
    print(f"export SE_SOLVER_GPUS={config.solver_gpus_str}")
    print(f"export SE_GEN_QUERY_GPUS={config.gen_query_gpus_str}")
    print(f"export SE_REWARD_PORTS={','.join(str(p) for p in config.reward_ports)}")
    print(f"export SE_N_CHALLENGER_GPUS={config.n_challenger_gpus}")
    print(f"export SE_N_REWARD_GPUS={config.n_reward_gpus}")
    print(f"export SE_N_SOLVER_GPUS={config.n_solver_gpus}")
    print(f"export SE_N_REWARD_SERVERS={config.n_reward_servers}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--shell":
        print_gpu_config_for_shell()
    else:
        config = get_gpu_config()
        print(config)

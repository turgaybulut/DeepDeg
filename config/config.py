from pathlib import Path
from typing import Optional

import yaml

from config.base_config import DeepDegConfig

_config: Optional[DeepDegConfig] = None

def load_config(config_path: str = "config.yaml") -> DeepDegConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    return DeepDegConfig(**config_data)

def get_config(config_path: str = "config.yaml") -> DeepDegConfig:
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config

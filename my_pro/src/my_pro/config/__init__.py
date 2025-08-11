import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_dir = Path(__file__).parent
    config_file = config_dir / f"{config_name}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_llm_config(llm_name: str) -> Dict[str, Any]:
    """Get LLM configuration by name."""
    llms_config = load_config("llms")
    if llm_name not in llms_config:
        raise KeyError(f"LLM configuration '{llm_name}' not found")
    return llms_config[llm_name]

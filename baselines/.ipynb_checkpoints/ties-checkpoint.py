import os
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from transformers import AutoTokenizer
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

@dataclass
class Config:
    config_dir: str
    result_dir: str
    lora_merge_cache: str
    base_model: str
    model_paths: List[str]
    weights: List[float]
    copy_tokenizer: bool
    lazy_unpickle: bool
    low_cpu_memory: bool

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    return Config(
        config_dir=config_dict['paths']['config_dir'],
        result_dir=config_dict['paths']['result_dir'],
        lora_merge_cache=config_dict['paths']['lora_merge_cache'],
        base_model=config_dict['models']['base_model'],
        model_paths=config_dict['models']['model_paths'],
        weights=config_dict['merge_options']['weights'],
        copy_tokenizer=config_dict['merge_options']['copy_tokenizer'],
        lazy_unpickle=config_dict['merge_options']['lazy_unpickle'],
        low_cpu_memory=config_dict['merge_options']['low_cpu_memory']
    )

def create_merge_config(weight: float, models: List[str], base_model: str) -> Dict[str, Any]:
    """Create merge configuration dictionary for a given weight."""
    return {
        "models": [{
            "model": models[i],
            "parameters": {
                "weight": weight if i == 0 else 1.0 - weight,
                "density": 1
            }
        } for i in range(len(models))],
        "merge_method": "ties",
        "base_model": base_model
    }

def generate_yaml_configs(config: Config) -> List[str]:
    """Generate YAML configuration files for different weights."""
    os.makedirs(config.config_dir, exist_ok=True)
    config_files = []

    for weight in config.weights:
        ties_dict = create_merge_config(weight, config.model_paths, config.base_model)
        filename = f"ties_weight_{weight}.yaml"
        filepath = os.path.join(config.config_dir, filename)
        
        with open(filepath, 'w') as file:
            yaml.dump(ties_dict, file, default_flow_style=False)
        
        config_files.append(filename)
    
    return config_files

def run_model_merges(config: Config, config_names: List[str]):
    """Run model merges for all configurations."""
    os.makedirs(config.result_dir, exist_ok=True)

    for config_name in sorted(config_names):
        config_path = os.path.join(config.config_dir, config_name)
        output_path = os.path.join(
            config.result_dir,
            ".".join(config_name.split(".")[:-1])
        )

        # Load and validate merge configuration
        with open(config_path, "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

        # Run merge operation
        run_merge(
            merge_config,
            out_path=output_path,
            options=MergeOptions(
                lora_merge_cache=config.lora_merge_cache,
                # cuda=torch.cuda.is_available(),
                cuda=False,
                copy_tokenizer=config.copy_tokenizer,
                lazy_unpickle=config.lazy_unpickle,
                low_cpu_memory=config.low_cpu_memory,
            ),
        )
        print(f"Completed merge for: {config_name}")

def save_tokenizers(config: Config, config_names: List[str]):
    """Save tokenizer for each merged model."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_paths[0])

    for config_name in config_names:
        output_path = os.path.join(
            config.result_dir,
            ".".join(config_name.split(".")[:-1])
        )
        tokenizer.save_pretrained(output_path)
        print(f"Saved tokenizer for: {config_name}")

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model merges based on configuration')
    parser.add_argument('--config', '-c', 
                      type=str, 
                      required=True,
                      help='Path to the YAML configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate YAML configurations
    config_names = generate_yaml_configs(config)
    print(f"Generated {len(config_names)} configuration files")
    
    # Run model merges
    run_model_merges(config, config_names)
    
    # Save tokenizers
    save_tokenizers(config, config_names)

if __name__ == "__main__":
    main()
import os
import yaml
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any, Dict, List
import torch
from src.ML_TEMPERATURE_PREDICTION.logging import logger

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox
    
    Args:
        path_to_yaml (Path): Path to YAML file
        
    Raises:
        ValueError: If YAML file is empty
        e: Empty file
        
    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e

def create_directories(path_to_directories: list, verbose=True):
    """
    Create directories
    
    Args:
        path_to_directories (list): List of paths
        verbose (bool, optional): Whether to log info. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

def save_json(path: Path, data: dict):
    """
    Save data as JSON file
    
    Args:
        path (Path): Path to JSON file
        data (dict): Data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"JSON file saved at: {path}")

def load_json(path: Path) -> ConfigBox:
    """
    Load JSON file
    
    Args:
        path (Path): Path to JSON file
        
    Returns:
        ConfigBox: ConfigBox object
    """
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

def save_model(path: Path, model: Any):
    """
    Save PyTorch model
    
    Args:
        path (Path): Path to save model
        model (Any): PyTorch model
    """
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    torch.save(model, path)
    logger.info(f"Model saved at: {path}")

def load_model(path: Path, device=None):
    """
    Load PyTorch model
    
    Args:
        path (Path): Path to model
        device: Device to load model to
        
    Returns:
        Any: PyTorch model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(path, map_location=device)
    logger.info(f"Model loaded from: {path}")
    return model

def save_binary(data: Any, path: Path):
    """
    Save binary file
    
    Args:
        data (Any): Data to save
        path (Path): Path to save data
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

def load_binary(path: Path) -> Any:
    """
    Load binary file
    
    Args:
        path (Path): Path to file
        
    Returns:
        Any: Object stored in file
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data
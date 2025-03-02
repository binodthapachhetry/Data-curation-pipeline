"""Configuration system for the data curation pipeline."""

import os
from typing import Any, Dict, List, Optional, Union

import yaml


class Config:
    """Configuration manager for the data curation pipeline.
    
    Handles loading, validating, and accessing configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize a new configuration.
        
        Args:
            config_path: Optional path to a YAML configuration file.
        """
        self.config: Dict[str, Any] = {}
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the configuration file is invalid YAML.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration settings.
        """
        self.config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: The configuration key to retrieve.
            default: Default value to return if the key doesn't exist.
            
        Returns:
            Any: The configuration value or the default.
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: The configuration key to set.
            value: The value to set.
        """
        self.config[key] = value
    
    def save(self, config_path: str) -> None:
        """Save the configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration file.
            
        Raises:
            yaml.YAMLError: If the configuration cannot be serialized to YAML.
        """
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """Validate the configuration against a schema.
        
        Args:
            schema: A dictionary describing the expected configuration structure.
            
        Returns:
            List[str]: A list of validation errors, empty if valid.
        """
        # Simple validation implementation
        errors = []
        
        for key, spec in schema.items():
            if key not in self.config:
                if spec.get('required', False):
                    errors.append(f"Missing required configuration key: {key}")
            else:
                value = self.config[key]
                expected_type = spec.get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}")
        
        return errors

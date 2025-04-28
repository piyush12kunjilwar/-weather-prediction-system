import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging
from .logger import log_error, log_info

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = None
        self.logger = logging.getLogger('rain_prediction')
        
    def _load_env_vars(self) -> Dict[str, str]:
        """Load environment variables with RAIN_PREDICTION_ prefix."""
        env_vars = {}
        for key, value in os.environ.items():
            if key.startswith('RAIN_PREDICTION_'):
                # Convert RAIN_PREDICTION_DATA_PATH to data.path
                config_key = key.replace('RAIN_PREDICTION_', '').lower().replace('__', '.')
                env_vars[config_key] = value
        return env_vars
        
    def _update_with_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with environment variables."""
        env_vars = self._load_env_vars()
        for key, value in env_vars.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        return config
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file and environment variables.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Update with environment variables
            self.config = self._update_with_env_vars(self.config)
            
            log_info(self.logger, "Configuration loaded successfully")
            return self.config
        except FileNotFoundError as e:
            log_error(self.logger, e, "Configuration file not found")
            raise
        except yaml.YAMLError as e:
            log_error(self.logger, e, "Invalid configuration file")
            raise
            
    def validate_config(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
            
        required_sections = ['data', 'model', 'logging', 'error_handling', 'paths']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
                
        # Validate paths
        for path in self.config['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)
            
        # Validate model configuration
        if 'architecture' not in self.config['model']:
            raise ValueError("Missing model architecture configuration")
            
        # Validate data configuration
        if 'raw_data_path' not in self.config['data']:
            raise ValueError("Missing raw data path configuration")
            
        log_info(self.logger, "Configuration validation successful")
        return True
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if not self.config:
            self.load_config()
        return self.config
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section (str): Section name
            
        Returns:
            Dict[str, Any]: Section configuration
            
        Raises:
            KeyError: If section doesn't exist
        """
        if not self.config:
            self.load_config()
        if section not in self.config:
            raise KeyError(f"Section {section} not found in configuration")
        return self.config[section]
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates (Dict[str, Any]): Dictionary of updates
            
        Raises:
            ValueError: If updates are invalid
        """
        if not self.config:
            self.load_config()
            
        def update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        self.config = update_dict(self.config, updates)
        self.validate_config()
        log_info(self.logger, "Configuration updated successfully")
        
    def save_config(self, path: str = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path (str, optional): Path to save configuration
            
        Raises:
            IOError: If save fails
        """
        if not self.config:
            raise ValueError("No configuration to save")
            
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
            log_info(self.logger, f"Configuration saved to {save_path}")
        except IOError as e:
            log_error(self.logger, e, "Failed to save configuration")
            raise 
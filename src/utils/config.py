"""Configuration manager with YAML loading and environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """Manage application configuration from YAML files with environment variable substitution."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration manager.

        Parameters
        ----------
        config_dict : Dict
            Configuration dictionary
        """
        self._config = config_dict

    @classmethod
    def load_config(cls, config_path: str = "config/config.yaml") -> "ConfigManager":
        """Load configuration from YAML file.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file (default: "config/config.yaml")

        Returns
        -------
        ConfigManager
            Configuration manager instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            # Return default configuration instead of failing
            print(f"Warning: Config file not found: {config_path}, using default configuration")
            return cls(cls._get_default_config())

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Substitute environment variables
        config_dict = cls._substitute_env_vars(config_dict)

        return cls(config_dict)

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration for when config file is missing.

        Returns
        -------
        Dict
            Default configuration dictionary
        """
        return {
            "data": {
                "sources": {"primary": "yfinance"},
                "fetch_settings": {"max_retries": 3, "retry_delay": 2.0, "timeout": 30.0},
                "validation": {
                    "max_missing_ratio": 0.10,  # Allow 10% missing for weekends/holidays
                    "outlier_zscore_threshold": 6.0,  # Less aggressive for real data
                    "price_jump_threshold": 0.25
                },
                "storage": {"raw_data_path": "data/raw", "processed_data_path": "data/processed"}
            },
            "cointegration": {
                "method": "johansen",
                "significance_level": 0.05,
                "johansen": {"deterministic_term": "c", "test_statistic": "trace"},
                "spread": {"lookback_window": 252, "half_life_min": 5, "half_life_max": 60}
            },
            "strategy": {
                "signals": {
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss": 4.0
                },
                "filters": {"min_holding_period": 5, "max_trades": 100}
            },
            "backtesting": {
                "initial_capital": 100000.0,
                "commission": 0.001,
                "slippage": 0.0005,
                "position_size": 0.20
            },
            "risk": {
                "var": {"confidence_level": 0.95, "time_horizon_days": 1},
                "position_limits": {"max_position_size": 0.30, "max_portfolio_var": 0.05}
            },
            "optimization": {
                "method": "bayesian",
                "n_iterations": 50,
                "n_initial_points": 10,
                "acquisition_function": "EI"
            }
        }

    @staticmethod
    def _substitute_env_vars(config: Any) -> Any:
        """Recursively substitute environment variables in config (${VAR_NAME} syntax).

        Parameters
        ----------
        config : Any
            Configuration value (dict, list, or str)

        Returns
        -------
        Any
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: ConfigManager._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigManager._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR_NAME} with environment variable
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, config)
            for var_name in matches:
                env_value = os.getenv(var_name, f"${{{var_name}}}")
                config = config.replace(f"${{{var_name}}}", env_value)
            return config
        else:
            return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation).

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.

        Returns
        -------
        Dict
            Configuration dictionary
        """
        return self._config.copy()

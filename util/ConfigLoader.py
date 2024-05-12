# /util/ConfigLoader.py

import os
import json
import yaml
from collections import OrderedDict
from ruamel.yaml import YAML

class _ConfigLoader:
    def __init__(self):
        self.config_path = './config/'
        env = os.getenv('ENV')
        self._config = self.load_default_config()
        self.override_config(env)

    def load_default_config(self):
        default_file = os.path.join(self.config_path, 'default.yaml') if os.path.exists(os.path.join(self.config_path, 'default.yaml')) else os.path.join(self.config_path, 'default.json')
        with open(default_file, 'r') as file:
            if default_file.endswith('.yaml'):
                return yaml.safe_load(file)
            else:
                return json.load(file)

    def override_config(self, name, path='./config/'):
        if name:
            config_file = os.path.join(path, f'{name}.yaml') if os.path.exists(os.path.join(path, f'{name}.yaml')) else os.path.join(path, f'{name}.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as file:
                    if config_file.endswith('.yaml'):
                        config = yaml.safe_load(file)
                    else:
                        config = json.load(file)
                self._config = self.deep_merge_dicts(self._config, config)

    @staticmethod
    def deep_merge_dicts(original, update):
        """
        Recursively merge two dictionaries.
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in original:
                original[key] = _ConfigLoader.deep_merge_dicts(original[key], value)
            else:
                original[key] = value
        return original

    def load_config(self, name, path='./config/'):
        self.override_config(name, path=path)
        return self

    def get(self):
        return self._config

class ConfigLoader:
    _instance = None
    
    yaml = YAML()
    yaml.preserve_quotes = True

    def __new__(cls, reload=False):
        if cls._instance is None or reload:
            cls._instance = _ConfigLoader()
        return cls._instance
    
    @staticmethod
    def read_config(name, path='./config/'):
        config = {}
        
        if name:
            config_file = os.path.join(path, name)
            if os.path.exists(config_file):
                with open(config_file, 'r') as file:
                    if config_file.endswith('.yaml'):
                        config = ConfigLoader.yaml.load(file)
                    else:
                        config = json.load(file, object_pairs_hook=OrderedDict)
                        
        return config
    
    @staticmethod
    def save_config(name, path='./config/', config={}):
        if name:
            # Construct the file path
            config_file = os.path.join(path, name)
            
            # Ensure the directory exists
            os.makedirs(path, exist_ok=True)
            
            # Open the file and save the config
            with open(config_file, 'w') as file:
                if config_file.endswith('.yaml'):
                    ConfigLoader.yaml.dump(config, file)
                else:
                    json.dump(config, file, indent=4, cls=OrderedDictEncoder)
                    
                    
# Custom JSON Encoder to maintain order in JSON
class OrderedDictEncoder(json.JSONEncoder):
    def encode(self, o):
        if isinstance(o, OrderedDict):
            return "{" + ", ".join(f'"{k}": {self.encode(v)}' for k, v in o.items()) + "}"
        else:
            return super().encode(o)

# /util/ConfigLoader.py

import os
import json
import yaml

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

    def override_config(self, name):
        if name:
            config_file = os.path.join(self.config_path, f'{name}.yaml') if os.path.exists(os.path.join(self.config_path, f'{name}.yaml')) else os.path.join(self.config_path, f'{name}.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as file:
                    if config_file.endswith('.yaml'):
                        config = yaml.safe_load(file)
                    else:
                        config = json.load(file)
                self._config.update(config)

    def load_config(self, name):
        self.override_config(name)
        return self

    def get(self):
        return self._config

class ConfigLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = _ConfigLoader()
        return cls._instance

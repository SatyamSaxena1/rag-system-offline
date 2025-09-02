class Config:
    def __init__(self, config_file='configs/default.yaml'):
        self.config_file = config_file
        self.settings = self.load_config()

    def load_config(self):
        import yaml
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value

    def save(self):
        import yaml
        with open(self.config_file, 'w') as file:
            yaml.dump(self.settings, file)

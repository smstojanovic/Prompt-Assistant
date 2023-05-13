import os
import json
import sys



class ConfigReader:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ConfigReader, cls).__new__(cls)
            cls._instance.config = {}
            cls._instance.load_config()
        return cls._instance
    
    def load_config(self):
        """
            Loads config from config in same directory as app entrypoint.
        """
        main_file_path = os.path.abspath(sys.argv[0])
        directory_path = os.path.dirname(main_file_path)
        app_env = os.getenv('APP_ENV', 'loc')
        config_file_name = f'recorder.{app_env}.cnfg.json'
        config_file = os.path.join(directory_path, config_file_name)
        
        try:
            with open(config_file, 'r') as file:
                self.config = json.load(file)
        except FileNotFoundError:
            print(f"Config file '{config_file}' not found.")
    
    def get_config(self):
        return self.config

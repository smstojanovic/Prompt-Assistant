import requests
from abc import ABC
from assistant.apps.recorder.utils.config_loader import ConfigReader
import json
cnfg = ConfigReader()

class TorchserveClient(ABC):
    MODEL_NAME = ''
    def __init__(self):
        config = cnfg.get_config()
        SERVER_HOST = config['torchserve']['host']
        SERVER_PORT = config['torchserve']['port']
        self.server_url = f"http://{SERVER_HOST}:{SERVER_PORT}/predictions/{self.MODEL_NAME}"
    
    def send_request(self, data):
        try:
            response = requests.post(self.server_url, json=data)
            if response.status_code == 200:
                result = response.json()
                # Process the result as needed
                return result
            else:
                print(f"Request failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

class JenkinsPromptClient(TorchserveClient):
    MODEL_NAME = 'jenkins_prompt'
    DTYPE = 'float32'

    def process_result(self, result):
        return result['transcribed_text']

    def do_inference(self, data, compressed):
        if type(data) != list:
            raise Exception('data needs to be list format')
        
        if type(compressed) != bool:
            raise Exception('compressed needs to be bool type')

        json_data = {
            'raw_data' : data,
            'dtype' : self.DTYPE,
            'compression' : compressed
        }

        result = self.send_request(json_data)
        if result is None:
            return None
        return self.process_result(result)
    
class JenkinsListenClient(JenkinsPromptClient):
    #MODEL_NAME = 'jenkins_listen'
    MODEL_NAME = 'jenkins_prompt'

class JenkinsPerceiveClient(JenkinsPromptClient):
    #MODEL_NAME = 'jenkins_listen'
    MODEL_NAME = 'jenkins_perceive'
    DTYPE = 'int16'

    def process_result(self, result):
        return result['speech_segments']

class JenkinsSpeechClient(TorchserveClient):
    MODEL_NAME = 'jenkins_speak'

    def do_inference(self, data):
        if type(data) != str:
            raise Exception('data needs to be a string')

        json_data = {
            'input_text' : data
        }

        result = self.send_request(json_data)
        if result is None:
            return None
        return result
    
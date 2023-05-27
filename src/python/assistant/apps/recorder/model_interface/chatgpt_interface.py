from collections import deque
import openai
from assistant.apps.recorder.utils.config_loader import ConfigReader
cnfg = ConfigReader()

OUTGOING_MESSAGE_PROMPT = """
This is a message coming from a user which has been transcribed via speech to text.
Try make the best sense out of it as you can and please make your response no more than 50 words.

The message is delimeted in quotes ``` as below:
```{incoming_prompt}```
"""

class ChatGPTInterface:
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0, history_len = 4):
        config = cnfg.get_config()
        api_key = config['openai']['api_key']
        openai.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.messages = deque(maxlen=2*history_len)
        self.loaded_message=False

    def append_message(self, role, message):
        if role == "user":
            message = OUTGOING_MESSAGE_PROMPT.format(incoming_prompt=message)

        self.messages.append(
            {"role": role, "content": message}
        )

    def get_prompt(self):
        return list(self.messages)

    def preload_user_message(self, incoming_text):
        self.loaded_message=True
        self.append_message('user', incoming_text)

    def chat(self, incoming_text=None):
        if not self.loaded_message:
            self.preload_user_message(incoming_text)

        prompt = self.get_prompt()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=prompt,
            temperature=self.temperature,
            #max_tokens=50,
            #n=1,
            #stop=None,
            #log_level="info"
        )

        reply = response.choices[0].message.content.strip()
        self.append_message('system', reply)
        self.loaded_message = False

        return reply

import itertools

CALL_NAMES= ['Gaia','Guy Uh']

INDICATORS = [
    'OK', 
    'Hey', 
    'Hi',
    'A',
    'Eh',
    'Hello',
    'Okay'
]

import string

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

class PromptDiscriminator:
    def __init__(self, indicators=INDICATORS, call_names=CALL_NAMES):
        self.indicators = [indicator.lower() for indicator in indicators]
        self.call_names = [call_name.lower() for call_name in call_names]
    
    def check_prompt(self, text):
        text_lower = remove_punctuation(text.lower())
        for indicator, call_name in itertools.product(self.indicators,self.call_names):
            if f'{indicator} {call_name}' in text_lower:
                return True
        return False

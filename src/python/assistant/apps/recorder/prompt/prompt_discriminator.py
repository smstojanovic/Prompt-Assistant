INDICATORS = [
    'OK Jenkins', 
    'Hey Jenkins', 
    'Hi Jenkins',
    'A Jenkins',
    'Eh Jenkins',
    'Hello Jenkins',
]

class PromptDiscriminator:
    def __init__(self, indicators=INDICATORS):
        self.indicators = [indicator.lower() for indicator in indicators]
    
    def check_prompt(self, text):
        text_lower = text.lower()
        for indicator in self.indicators:
            if indicator in text_lower:
                return True
        return False

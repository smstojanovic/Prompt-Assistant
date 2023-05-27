try:
    from audio_processors.text_synthesizer import TextSynthesizeHandler
except:
    # loads inside torchserve in the same directory
    from text_synthesizer import TextSynthesizeHandler

import json

class MainSpeechSynthesizer(TextSynthesizeHandler):
    pass

_service = MainSpeechSynthesizer()

# need this to compile the served model

def handler(data, context):
    """
        Handles text syenthesizing.
    """

    try:
        if not _service.initialised:
            #voice_embedding = voice_embedding['voice_embedding']
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e

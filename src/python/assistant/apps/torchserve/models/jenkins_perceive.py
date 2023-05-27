try:
    from audio_processors.audio_vad import AudioVADHandler
except:
    # loads inside torchserve in the same directory
    from audio_vad import AudioVADHandler

class PromptVADHandler(AudioVADHandler):
    pass

_service = PromptVADHandler()

# need this to compile the served model
def handler(data, context):
    try:
        if not _service.initialised:
            context.system_properties['BASE_MODEL'] = 'base.en'
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e

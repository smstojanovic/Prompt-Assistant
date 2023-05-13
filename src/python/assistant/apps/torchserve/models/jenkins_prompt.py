try:
    from audio_processors.audio_transcription import AudioTranscriptionHandler
except:
    # loads inside torchserve in the same directory
    from audio_transcription import AudioTranscriptionHandler


class ListenerTranscriptionHandler(AudioTranscriptionHandler):
    pass

_service = ListenerTranscriptionHandler()

# need this to compile the served model

def handler(data, context):
    """
        For now this is the same as jenkins_listen.
        The ideal state is to go to much smaller and faster models
        that only return true or false whether an incoming 'audio signal' (maybe it doesn't originate
        from a traditional microphone, but rather the analog output of some arbitrary sensor) hits the
        requirements for prompts to activate.
        
        The larger audio transcription model mixed with recording new signals can be used to train such a 
        smaller model in the future.
    """
    try:
        if not _service.initialised:
            context.system_properties['BASE_MODEL'] = 'tiny.en'
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e

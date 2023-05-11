from abc import ABC
import torch
import logging
import numpy as np

import whisper
from ts.torch_handler.base_handler import BaseHandler

BASE_MODEL = 'tiny.en'
DTYPE_MAP = { 'int8': np.int8, 'int16': np.int16, 'int32': np.int32 }


logger = logging.getLogger(__name__)

class AudioTranscriptionHandler(BaseHandler, ABC):
    def _init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        # self.manifest=  ctx.manifest
        # properties = ctx.system_properties
        # model_dir = properties.get('model_dir')
        self.device = 'cuda:0'
        self.model = whisper.load_model(BASE_MODEL)

        #self.model.to(self.device)
        #self.model.eval()

        logger.debug('loaded model')
        self.initialized = True

    def preprocess(self, data):
        audio = data.get('raw_data')
        dtype = data.get('dtype')
        compression = data.get('compression')
        
        audio = np.array(audio, dtype=DTYPE_MAP[dtype])

        if compression:
            # uncompress. handle this later.
            pass

        audio = whisper.pad_or_trim(audio)

        return audio

    def inference(self, inputs):
        with torch.no_grad():
            prediction = self.model.transcribe(inputs)

        prediction = torch.mean(prediction[0], dim=1)
        prediction = prediction["text"]

        return [prediction]

    def postprocess(self, inference_output):
        output = []
        for text in inference_output:
            output.append({'transcribed_text' : text})

        return output

_service = AudioTranscriptionHandler()

# need this to compile the served model
def handler(data, context):
	try:
		if not _service.initialised:
			_service.initialize(context)

		if data is None:
			return None

		data = _service.preprocess(data)
		data = _service.inference(data)
		data = _service.postprocess(data)

		return data
	except Exception as e:
		raise e

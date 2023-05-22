from abc import ABC
import torch
import logging
import numpy as np

from speechbrain.pretrained import VAD
from ts.torch_handler.base_handler import BaseHandler

BASE_MODEL = 'speechbrain/vad-crdnn-libriparty'
DTYPE_MAP = { 'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'float32' : np.float32 }


logger = logging.getLogger(__name__)

class AudioVADHandler(BaseHandler, ABC):
    def _init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        arch = properties.get('BASE_MODEL',BASE_MODEL)
        #model_dir = properties.get('model_dir')
        self.device = 'cuda:0'
        self.model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty", 
            run_opts={"device":self.device}
        )

        #self.model.to(self.device)
        self.model.eval()

        logger.debug('loaded model')
        self.initialized = True

    def preprocess(self, data):
        #print(data)
        if type(data) == list:
            data = data[0]['body']

        audio = data.get('raw_data')
        dtype = data.get('dtype')
        compression = data.get('compression')

        audio_data = np.array(audio, dtype=DTYPE_MAP.get(dtype, np.float32))
        audio_data = audio_data.flatten() / 32768.0

        # if compression:
        #     # uncompress. handle this later.
        #     pass

        return audio_data

    def inference(self, inputs):
        with torch.no_grad():
            #mel = whisper.log_mel_spectrogram(inputs).to(self.model.device)
            prediction = self.model.transcribe(inputs)

        prediction = prediction["text"]

        return [prediction]

    def postprocess(self, inference_output):
        output = []
        for text in inference_output:
            output.append({'transcribed_text' : text})

        return output

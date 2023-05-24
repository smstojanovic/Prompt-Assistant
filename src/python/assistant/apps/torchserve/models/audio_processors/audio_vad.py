from abc import ABC
import torch
import logging
import numpy as np

from ts.torch_handler.base_handler import BaseHandler

from io import BytesIO
import wave

import torch
import torchaudio
from speechbrain.pretrained import split_path, fetch, VAD

try:
    from assistant.apps.torchserve.models.modifications.vad import InMemoryVAD
except:
    # loads inside torchserve in the same directory
    from vad import InMemoryVAD

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
        self.model = InMemoryVAD.from_hparams(
            source=BASE_MODEL,
            run_opts={"device":self.device}
        )
        self.model.device = self.device

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

        bytes_io = BytesIO()

        with wave.open(bytes_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(np.array(audio, dtype=DTYPE_MAP.get(dtype, np.int16)))

        bytes_io.seek(0)

        # if compression:
        #     # uncompress. handle this later.
        #     pass

        return bytes_io

    def inference(self, inputs):
        with torch.no_grad():
            #mel = whisper.log_mel_spectrogram(inputs).to(self.model.device)
            speech_segments = self.model.get_speech_segments(inputs)

        speech_segments = speech_segments.to('cpu')
        
        prediction = speech_segments.numpy().tolist()

        return [prediction]

    def postprocess(self, inference_output):
        output = []
        for text in inference_output:
            output.append({'speech_segments' : text})

        return output

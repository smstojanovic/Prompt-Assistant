from abc import ABC
import torch
import logging
import numpy as np
import json

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from ts.torch_handler.base_handler import BaseHandler
from datasets import load_dataset

BASE_MODEL = 'microsoft/speecht5_tts'
VOCODER_MODEL = 'microsoft/speecht5_hifigan'

logger = logging.getLogger(__name__)

class TextSynthesizeHandler(BaseHandler, ABC):
    def _init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        #properties = ctx.system_properties
        
        # with open('voice_embeddings.json','r') as f:
        #     voice_embedding = json.load(f)

        # voice_embedding = voice_embedding['voice_embedding']
        self.device = 'cuda:0'

        self.processor = SpeechT5Processor.from_pretrained(BASE_MODEL)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(BASE_MODEL)
        self.vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_MODEL)

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

        # move to device
        self.model.to(self.device)
        self.vocoder.to(self.device)

        #eval models
        self.model.eval()
        self.vocoder.eval()

        # embeddings
        #self.speaker_embeddings = torch.tensor(voice_embedding, dtype=torch.float32).reshape(1,512)
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        self.speaker_embeddings = self.speaker_embeddings.to(self.device)

        logger.debug('loaded model')
        self.initialized = True

    def preprocess(self, data):
        #print(data)
        if type(data) == list:
            data = data[0]['body']

        input_text = data.get('input_text')

        inputs = self.processor(text=input_text, return_tensors="pt")
        inputs = inputs.to('cuda:0')

        return inputs

    def inference(self, inputs):
        spectrogram = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings)
        
        with torch.no_grad():
            speech = self.vocoder(spectrogram)

        speech = speech.to('cpu')
        speech = speech.numpy()
        speech = speech.tolist()

        return [speech]

    def postprocess(self, inference_output):
        output = []
        for speech in inference_output:
            output.append({
                'speech' : speech,
                'sample_rate' : 16000,
                })

        return output

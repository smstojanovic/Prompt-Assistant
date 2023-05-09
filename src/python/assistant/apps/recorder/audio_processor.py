import numpy as np
import threading
import multiprocessing
from assistant.libs.buffers.fixed_size_buffer import FixedAudioBuffer
from assistant.libs.compression.audio_compressor import FLACAudioCompressor
import time
import librosa

class AudioProcessor:
    def __init__(self, sample_rate:int, process_interval_seconds:float = 0.5, compress=True):
        self.processing_enabled = True
        self.sample_rate = sample_rate
        self.process_interval_seconds = process_interval_seconds
        
        # depending on network and processing power of the device, it may or may not be worthwhile
        # compressing the audio before it goes to the model. Will need to test this.
        self.do_compress=True

    def stop_processing(self):
        """
            if record_audio is called, calling this will stop it.
        """
        self.processing_enabled = False

    def process_audio(self, audio_buffer: FixedAudioBuffer, use_thread:bool=True):
        """
            Processes audio.
        """

        if use_thread:
            # Run in a new thread
            t = threading.Thread(target=self._process_audio, args=(audio_buffer, ))
            t.start()
            return t
        else:
            # Run in a new process
            p = multiprocessing.Process(target=self._process_audio, args=(audio_buffer, ))
            p.start()
            return p
        
    def _process_audio(self, audio_buffer: FixedAudioBuffer, target_sample_rate=16000):
        """
            Periodically get the buffer, resample and compress it.
            Wrap it up and send it to ASR model to transcribe.
        """
        while self.processing_enabled:

            time.sleep(self.process_interval_seconds)

            audio = audio_buffer.get()
            resampled = librosa.resample(audio, self.sample_rate, target_sample_rate)
            
            # make this verbose. Either we compress it or apply the identity compression, 
            # either way, we will send 'compressed' to our transcription model
            if self.do_compress:
                compressor = FLACAudioCompressor()
                compressed = compressor.compress(resampled)
            else:
                compressed = resampled


            # send to transcription model
            x = 1

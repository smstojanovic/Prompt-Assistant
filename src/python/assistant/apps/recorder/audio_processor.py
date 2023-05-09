import numpy as np
import threading
import multiprocessing
from assistant.libs.buffers.fixed_size_buffer import FixedAudioBuffer
from assistant.libs.compression.audio_compressor import FLACAudioCompressor
import time
import librosa
import samplerate

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

            # wait till the buffer is full
            if audio_buffer.write_index < audio_buffer.buffer_size:
                continue

            audio = audio_buffer.get()
            # if there's an infinite value don't send
            if np.any(~np.isfinite(audio)):
                continue

            audio = audio.astype(np.float32)

            resampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sample_rate)
            # resampling_ratio = target_sample_rate / self.sample_rate
            # resampler = samplerate.Resampler('sinc_best')
            # resampled = resampler.process(audio, resampling_ratio)
             
            
            # make this verbose. Either we compress it or apply the identity compression, 
            # either way, we will send 'compressed' to our transcription model
            if self.do_compress:
                compressor = FLACAudioCompressor(target_sample_rate)
                compressed = compressor.compress(resampled)
            else:
                compressed = resampled


            # send to transcription model
            x = 1


# import wave
# # save to wave file.
# wav_output_filename = 'test2.wav'
# audio_np = compressed
# SAMPLE_RATE = 48000
# #audio_bytes = wave.struct.pack("<" + str(len(audio_np)) + "h", *audio_np)

# with wave.open(wav_output_filename, 'wb') as wav_file:
#     wav_file.setnchannels(1)
#     wav_file.setsampwidth(2)
#     wav_file.setframerate(16000)
#     wav_file.writeframes(audio_np)

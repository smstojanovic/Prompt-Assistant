import numpy as np
import threading
import multiprocessing
from assistant.libs.buffers.fixed_size_buffer import FixedAudioBuffer
from assistant.libs.compression.audio_compressor import FLACAudioCompressor
from assistant.apps.recorder.ts_interface.ts_client import JenkinsPromptClient, JenkinsListenClient
from assistant.apps.recorder.prompt.prompt_discriminator import PromptDiscriminator
from assistant.libs.buffers.audio_buffer_handler import BufferMode
import time
from datetime import datetime
import librosa
import samplerate

from assistant.apps.recorder.utils.config_loader import ConfigReader
cnfg = ConfigReader()

class AudioProcessor:
    def __init__(self, sample_rate:int, dtype, process_interval_seconds:float = 0.5, compress=True):
        self.processing_enabled = True
        self.sample_rate = sample_rate
        self.process_interval_seconds = process_interval_seconds
        self.dtype=dtype
        
        # depending on network and processing power of the device, it may or may not be worthwhile
        # compressing the audio before it goes to the model. Will need to test this.
        self.do_compress=True
        self.prompt_client = JenkinsPromptClient()
        self.listen_client = JenkinsListenClient()
        self.prompt_discriminator = PromptDiscriminator()

        self.mode_process_map = {
            BufferMode.PROMPT : self._process_prompt,
            BufferMode.LISTEN : self._process_listen,
            BufferMode.SILENCE : self._process_silence
        }


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

    def _reshape_audio(self, audio_buffer: FixedAudioBuffer, target_sample_rate=16000, wait_for_ready = True):
        """
            Grabs the data in the audio buffer, resamples it to the target rate
            and potentially compresses it if we want it to.
        """
        # wait till the buffer is full / ready
        if wait_for_ready and not audio_buffer.ready():
            return None

        audio = audio_buffer.get()
        # if there's an infinite value don't send
        if np.any(~np.isfinite(audio)):
            return None

        audio = audio.astype(np.float32)

        resampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sample_rate)
        resampled = resampled.astype(self.dtype)
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
        return resampled

    def _process_audio(self, audio_buffer: FixedAudioBuffer, target_sample_rate=16000):
        """
            Periodically get the buffer, resample and compress it.
            Wrap it up and send it to ASR model to transcribe.
        """
        while self.processing_enabled:

            time.sleep(self.process_interval_seconds)
            # route based on buffer mode
            self.mode_process_map[audio_buffer.mode](audio_buffer, target_sample_rate)


    def _process_prompt(self, audio_buffer, target_sample_rate):
        loop_start_time = datetime.now()

        resampled = self._reshape_audio(audio_buffer, target_sample_rate)
        if resampled is None:
            return None

        # send to transcription model
        loop_inference_time = datetime.now()
        processing_time = loop_inference_time - loop_start_time

        # if the loop time is fast enough and everything is warmed up; start sending to model.
        if processing_time.seconds == 0 and processing_time.microseconds < 1e5:
            prompt_data = resampled.tolist()
            transcribed_speech = self.prompt_client.do_inference(prompt_data, self.do_compress)
            is_prompt = self.prompt_discriminator.check_prompt(transcribed_speech)
            if is_prompt:
                audio_buffer.set_mode(BufferMode.LISTEN)
                audio_buffer.reset()
                print('Listening...')


    def _process_listen(self, audio_buffer, target_sample_rate):
        loop_start_time = datetime.now()

        if audio_buffer.is_enabled() and audio_buffer.time_written() < 5:
            # still listening, wait till its not enabled
            return
            
        resampled = self._reshape_audio(audio_buffer, target_sample_rate, wait_for_ready=False)
        if resampled is None:
            return None

        # send to transcription model
        loop_inference_time = datetime.now()
        processing_time = loop_inference_time - loop_start_time

        # if the loop time is fast enough and everything is warmed up; start sending to model.
        if processing_time.seconds == 0 and processing_time.microseconds < 1e5:
            prompt_data = resampled.tolist()
            transcribed_speech = self.listen_client.do_inference(prompt_data, self.do_compress)
            #is_prompt = self.prompt_discriminator.check_prompt(transcribed_speech)
            if transcribed_speech:
                print('Prompting...')
                audio_buffer.reset()
                audio_buffer.set_mode(BufferMode.PROMPT)

    def _process_silence(self, audio_buffer, target_sample_rate):

        time.sleep(1)
        audio_buffer.set_mode(BufferMode.PROMPT)

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

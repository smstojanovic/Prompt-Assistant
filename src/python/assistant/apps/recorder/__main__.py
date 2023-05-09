import pyaudio
import numpy as np
from assistant.libs.buffers.fixed_size_buffer import FixedAudioBuffer
from assistant.apps.recorder.audio_collector import AudioCollector
from assistant.apps.recorder.audio_processor import AudioProcessor
import wave

CHUNK_SIZE = 9600
RECORDING_SECONDS = 3
SAMPLE_RATE = 48000
USE_THREAD = True # uses threads or processes

# Initialize fixed buffer
audio_buffer = FixedAudioBuffer(RECORDING_SECONDS, SAMPLE_RATE)
audio_collector = AudioCollector()
audio_processor = AudioProcessor()

audio_thread = audio_collector.record_audio(audio_buffer, SAMPLE_RATE, CHUNK_SIZE, use_thread=USE_THREAD)
processor_thread = audio_processor.process_audio(audio_buffer, use_thread=USE_THREAD)


audio_thread.join()
processor_thread.join()

audio_collector.stop_recording()
#audio_processor.stop_processing()
import wave
# save to wave file.
wav_output_filename = 'test1.wav'
audio_np = audio_buffer.get()
SAMPLE_RATE = 48000
#audio_bytes = wave.struct.pack("<" + str(len(audio_np)) + "h", *audio_np)

with wave.open(wav_output_filename, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)
    wav_file.writeframes(audio_np)

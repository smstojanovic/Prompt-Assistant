import pyaudio
import numpy as np
from assistant.libs.buffers.fixed_size_buffer import FixedAudioBuffer
import wave

CHUNK_SIZE = 4096
RECORDING_SECONDS = 3
SAMPLE_RATE = 48000

# Initialize ring buffer
audio_buffer = FixedAudioBuffer(RECORDING_SECONDS, SAMPLE_RATE)

# Initialize PyAudio
audio = pyaudio.PyAudio()

for i in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(i)
    print((i, dev['name'], dev['maxInputChannels']), dev['defaultSampleRate'])    

#format = pyaudio.paFloat32
format = pyaudio.paInt16
channels = 1
dev_index = 1

stream = audio.open(format=format,
                    channels=channels,
                    rate=SAMPLE_RATE,
                    input_device_index = dev_index,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

# Start streaming audio
print("Recording started")

while True:
    # Read audio data from stream
    try:
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow = False)
        audio_buffer.push(audio_data)
    except Exception as ex:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        break
    # Process audio data or do other logic here
    # ...

# save to wave file.
wav_output_filename = 'test1.wav'
audio_np = audio_buffer.get()
#audio_bytes = wave.struct.pack("<" + str(len(audio_np)) + "h", *audio_np)

with wave.open(wav_output_filename, 'wb') as wav_file:
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)
    wav_file.writeframes(audio_np)

# Stop streaming audio
stream.stop_stream()
stream.close()
audio.terminate()

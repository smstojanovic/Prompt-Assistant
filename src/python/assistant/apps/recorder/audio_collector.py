import pyaudio
import threading
import multiprocessing
import numpy as np
from assistant.libs.buffers.fixed_size_buffer import FixedAudioBuffer


class AudioCollector:
    def __init__(self, dtype):
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # whether to enable recording. setting to true for now, false will stop recording in 'record_audio' method.
        self.recording_enabled = True
        self.dtype=dtype

    def get_device_details(self):
        # Check device details.
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            print((i, dev['name'], dev['maxInputChannels']), dev['defaultSampleRate'])

    def stop_recording(self):
        """
            if record_audio is called, calling this will stop it.
        """
        self.running_allowed = False

    def record_audio(self, audio_buffer: FixedAudioBuffer, sample_rate:int, chunk_size:int, format = pyaudio.paInt16, use_thread:bool=True):
        """
            Records audio in either a thread or process.
        """

        #format = pyaudio.paFloat32
        # need to parameterise this later.
        # format = pyaudio.paInt16
        #format = pyaudio.paInt32
        channels = 1
        dev_index = 1

        if use_thread:
            # Run in a new thread
            t = threading.Thread(target=self._record_audio, args=(audio_buffer, sample_rate, chunk_size, format, channels, dev_index))
            t.start()
            return t
        else:
            # Run in a new process
            p = multiprocessing.Process(target=self._record_audio, args=(audio_buffer, sample_rate, chunk_size, format, channels, dev_index))
            p.start()
            return p

    def _record_audio(self, audio_buffer: FixedAudioBuffer, sample_rate:int, chunk_size:int, format, channels:int, dev_index:int):
        """
            Internal function for recording audio.
        """
        self.running_allowed = True
        stream = self.audio.open(format=format,
                            channels=channels,
                            rate=sample_rate,
                            input_device_index = dev_index,
                            input=True,
                            frames_per_buffer=chunk_size)

        # Start streaming audio
        print("Recording started")

        while self.running_allowed:
            # Read audio data from stream
            try:
                audio_data = stream.read(chunk_size, exception_on_overflow = False)
                # Convert the audio data to a NumPy array
                audio_samples = np.frombuffer(audio_data, dtype=self.dtype)
                # Remove non-finite values from the audio data
                audio_samples_finite = audio_samples[np.isfinite(audio_samples)]
                # Push the audio data into the buffer
                audio_buffer.push(audio_samples_finite)
            except Exception as ex:
                print("Recording ended")
                stream.stop_stream()
                stream.close()
                self.audio.terminate()
                break
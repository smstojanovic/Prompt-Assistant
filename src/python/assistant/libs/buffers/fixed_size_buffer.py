import numpy as np
from assistant.libs.buffers.audio_utils import save_audio

class FixedSizeBuffer:
    """
        Creates a buffer with a fixed size. Any chunk that will overflow the buffer will push older data out. 
        This is meant for a continuous audio stream so we can poll that last N seconds from a microphone reading.
    """
    def __init__(self, buffer_size, dtype=np.int16):
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.buffer = np.zeros(buffer_size, dtype=dtype)
        self.write_index = 0

    def push(self, data):
        # Convert data to numpy array
        audio_samples = np.frombuffer(data, dtype=self.dtype)

        # Determine how much space is available in the buffer
        space_left = self.buffer_size - self.write_index

        if len(audio_samples) <= space_left:
            # There is enough space in the buffer to write the new data
            self.buffer[self.write_index:self.write_index+len(audio_samples)] = audio_samples
            self.write_index += len(audio_samples)
        else:
            # There is not enough space in the buffer to write the new data
            rollsize = len(audio_samples) - space_left
            self.buffer = np.roll(self.buffer,-1*rollsize)
            self.buffer[-1*len(audio_samples):] = audio_samples
            self.write_index = self.buffer_size

    def get(self, num_samples=None):
        # If num_samples is greater than buffer size or none, return the entire buffer
        if num_samples is None or num_samples >= self.buffer_size:
            space_left = self.buffer_size - self.write_index
            if space_left > 0:
                return self.buffer[:self.write_index]
            return self.buffer

        # Copy samples from buffer to output array
        samples_tail = (self.write_index - num_samples) % self.buffer_size
        if samples_tail < self.write_index:
            return self.buffer[samples_tail:self.write_index]
        else:
            samples = np.zeros(num_samples, dtype=self.dtype)
            samples[:self.buffer_size - samples_tail] = self.buffer[samples_tail:]
            samples[self.buffer_size - samples_tail:self.write_index - samples_tail] = self.buffer[:self.write_index - samples_tail]
            return samples
        
    def ready(self):
        """
            Used for prompting the listener. Only start working when buffer is full
        """
        return self.write_index >= self.buffer_size
    
    def reset(self):
        self.buffer = np.zeros(self.buffer_size, dtype=self.dtype)
        self.write_index = 0

    def is_enabled(self, **kwargs):
        return True

class FixedAudioBuffer(FixedSizeBuffer):
    """
        Creates a buffer with a fixed size for Audio. Any chunk that will overflow the buffer will push older data out. 
        inputs

        Parameters:
            buffer_time_seconds (int): Buffer time in seconds specifies the length of time to hold onto audio.
            sample_rate_hz (int): Sample rate of incoming audio.

    """
    def __init__(self, buffer_time_seconds: int, sample_rate_hz: int, dtype=np.int16):
        bytes_per_element = dtype().itemsize
        self.sample_rate_hz = sample_rate_hz
        buffer_size = int(buffer_time_seconds*sample_rate_hz/bytes_per_element*2)
        super().__init__(buffer_size, dtype)

    def time_written(self):
        return 3
    
    def save(self, filename):
        save_audio(self.get(), filename, self.sample_rate)
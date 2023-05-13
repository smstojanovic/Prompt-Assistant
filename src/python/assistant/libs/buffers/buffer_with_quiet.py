import numpy as np

class BufferWithQuiet:
    """
    Creates a buffer with a fixed size for Audio. Any chunk that will overflow the buffer will push older data out. 
    It also monitors the incoming audio for quiet periods and disables push activity accordingly.

    Parameters:
        buffer_size (int): Buffer size specifies the total buffer size.
        quiet_buffer_size (int): The size of the quiet buffer, determines the time period for silence detection.
        dtype (numpy type): The data type of the buffer, typically an audio data type.
        quiet_threshold (float): The threshold of silence, expressed as a multiplier of the maximum amplitude.
    """
    def __init__(self, buffer_size, quiet_buffer_size, dtype=np.int16, quiet_threshold = 0.1):
        if buffer_size < quiet_buffer_size:
            raise Exception('Listening Buffer needs to be bigger than Quiet Buffer')
        
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.buffer = np.zeros(buffer_size, dtype=dtype)
        self.write_index = 0

        self.quiet_threshold = quiet_threshold

        # Variables for managing buffer state
        self.enabled = True
        self.max_amplitude = 0

        # Buffer to hold recent audio for quiet detection
        self.quiet_buffer_size = quiet_buffer_size
        self.quiet_index = 0

    def push(self, data):
        if not self.enabled:
            return

        # Convert data to numpy array
        audio_samples = np.frombuffer(data, dtype=self.dtype)

        # Update maximum amplitude
        self.max_amplitude = max(self.max_amplitude, np.max(np.abs(audio_samples)))

        # Write the data to the buffer
        end_index = self.write_index + len(audio_samples)
        if end_index <= self.buffer_size:
            self.buffer[self.write_index:end_index] = audio_samples
            self.write_index = end_index
        else:
            # Buffer is full, disable writing
            self.enabled = False
            return

        # Update the quiet buffer index
        self.quiet_index = max(0, self.write_index - self.quiet_buffer_size)

        # Check if the recent audio is below the quiet threshold
        try:
            main_rms = np.sqrt(np.mean(self.buffer[:self.quiet_index]**2))
        except:
            main_rms = self.max_amplitude
        quiet_rms = np.sqrt(np.mean(self.buffer[self.quiet_index:self.write_index]**2))
        
        if quiet_rms < main_rms * self.quiet_threshold and self.quiet_index >  0:
            self.enabled = False
            return

    def get(self, num_samples=None):
        # Exclude the last quiet part of audio
        return self.buffer[:self.quiet_index]

    def reset(self):
        self.buffer = np.zeros(self.buffer_size, dtype=self.dtype)
        self.write_index = 0
        self.quiet_index = 0
        self.enabled = True
        self.max_amplitude = 0

    def ready(self):
        return self.write_index > 0

    def is_enabled(self):
        return self.enabled


class AudioBufferWithQuiet(BufferWithQuiet):
    def __init__(self, buffer_time_seconds: int, sample_rate_hz: int, dtype=np.int16, quiet_seconds: int = 2, quiet_threshold: float = 0.1):
        bytes_per_element = dtype().itemsize
        self.bytes_per_element = bytes_per_element
        buffer_size = int(buffer_time_seconds*sample_rate_hz/bytes_per_element*2)
        quiet_size = int(quiet_seconds*sample_rate_hz/bytes_per_element*2)
        self.sample_rate = sample_rate_hz
        super().__init__(buffer_size, quiet_size, dtype, quiet_threshold)

    def time_written(self):
        return self.write_index/self.sample_rate*self.bytes_per_element/2

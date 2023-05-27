from enum import Enum
from assistant.libs.buffers.buffer_with_quiet import AudioBufferWithQuiet
from assistant.libs.buffers.fixed_size_buffer import FixedAudioBuffer
from assistant.libs.buffers.audio_utils import save_audio

class BufferMode(Enum):
    SILENCE = 1
    PROMPT = 2
    LISTEN = 3

class SilenceBuffer:
    """
    A silence buffer that holds nothing.
    """
    def push(self, data):
        pass

    def get(self, num_samples=None):
        return None

    def reset(self):
        pass

    def ready(self):
        """
            The silence buffer was born ready.
        """
        return True
    
    def time_written(self):
        return 0
    
    def save_audio(self, filename):
        return
    
    def is_enabled(self, **kwargs):
        return False

class AudioBufferHandler:
    """
    Handles two buffers, a FixedAudioBuffer and an AudioBufferWithQuiet, and switches between them based on a mode.
    """
    def __init__(self, prompt_buffer: FixedAudioBuffer, listen_buffer: AudioBufferWithQuiet):
        self.buffers = {
            BufferMode.SILENCE: SilenceBuffer(),
            BufferMode.PROMPT: prompt_buffer,
            BufferMode.LISTEN: listen_buffer
        }
        self.mode = BufferMode.PROMPT

    def set_mode(self, mode: BufferMode):
        if mode not in BufferMode:
            raise ValueError("Invalid mode")
        self.mode = mode
        if mode == BufferMode.SILENCE:
            self.reset()

    def push(self, data):
        self.buffers[self.mode].push(data)

    def get(self, num_samples=None):
        return self.buffers[self.mode].get(num_samples)

    def is_enabled(self, **kwargs):
        return self.buffers[self.mode].is_enabled(**kwargs)

    def ready(self):
        return self.buffers[self.mode].ready()
    
    def time_written(self):
        return self.buffers[self.mode].time_written()

    def reset(self):
        for buffer in self.buffers.values():
            buffer.reset()

    def save(self, filename):
        return self.buffers[self.mode].save(filename)

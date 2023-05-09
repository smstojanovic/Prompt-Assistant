import abc
import io
import numpy as np
import soundfile as sf
from enum import Enum

class CompressionTypes(Enum):
    FLAC='FLAC'


class AudioCompressor(abc.ABC):
    """
        For in-memory audio compression from numpy arrays
    """
    COMPRESSOR = None
    
    @abc.abstractmethod
    def compress(self, audio):
        pass


class FLACAudioCompressor(AudioCompressor):
    COMPRESSOR = CompressionTypes.FLAC

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    
    def compress(self, audio):
        """
            Compresses an audio signal as a FLAC-encoded numpy array.
        """
        compressed_audio = io.BytesIO()
        sf.write(compressed_audio, audio, self.sample_rate, format='FLAC')
        compressed_audio_bytes = compressed_audio.getvalue()
        compressed_audio_np = np.frombuffer(compressed_audio_bytes, dtype=np.uint8)
        return compressed_audio_np

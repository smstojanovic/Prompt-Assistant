import numpy as np
from numpy.testing import assert_array_equal
import pytest
import soundfile as sf
import io
from assistant.libs.compression.audio_compressor import FLACAudioCompressor


@pytest.fixture
def compressor():
    return FLACAudioCompressor(sample_rate=16000)


def test_flac_audio_compressor_compress(compressor):
    # Generate some test data
    num_samples = 3 * 16000
    audio = np.random.rand(num_samples).astype(np.float32)

    # Compress the test data
    compressed_audio = compressor.compress(audio)

    # Check that the compressed audio is a numpy array
    assert isinstance(compressed_audio, np.ndarray)

    # Check that the compressed audio is not empty
    assert compressed_audio.size > 0

    # Check that the compressed audio can be decoded
    decompressed_audio, sr = sf.read(io.BytesIO(compressed_audio.tobytes()))
    assert sr == compressor.sample_rate

    # Check that the decompressed audio matches the original audio (with tolerance)
    assert np.allclose(audio, decompressed_audio, rtol=1e-3, atol=1e-3)
    #assert_array_equal(audio, decompressed_audio)
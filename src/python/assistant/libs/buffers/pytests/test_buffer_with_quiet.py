import pytest
import numpy as np

from assistant.libs.buffers.buffer_with_quiet import BufferWithQuiet

def test_push_and_get():
    # Initialize a buffer with space for 10000 samples
    buffer = BufferWithQuiet(10000, 2000, dtype=np.float32, quiet_threshold=0.1)

    # Push 5000 samples of noise with amplitude 1.0
    noise = np.random.normal(0, 1.0, 5000).astype(np.float32)
    buffer.push(noise.tobytes())

    # Push 3000 samples of noise with amplitude 0.05 (quieter)
    quiet_noise = np.random.normal(0, 0.05, 3000).astype(np.float32)
    buffer.push(quiet_noise.tobytes())

    # Push 2000 samples of noise with amplitude 0.01 (even quieter)
    very_quiet_noise = np.random.normal(0, 0.01, 2000).astype(np.float32)
    buffer.push(very_quiet_noise.tobytes())

    # The RMS of the buffer's content should be close to the RMS of the initial noise
    buffer_content = buffer.get()
    rms_buffer = np.sqrt(np.mean(buffer_content**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    assert np.isclose(rms_buffer, rms_noise, rtol=0.1)

def test_buffer_disabled():
    # Initialize a buffer with space for 10000 samples
    buffer = BufferWithQuiet(10000, 2000, dtype=np.float32, quiet_threshold=0.1)

    # Push 5000 samples of noise with amplitude 1.0
    noise = np.random.normal(0, 1.0, 5000).astype(np.float32)
    buffer.push(noise.tobytes())

    # Push 1000 samples of noise with amplitude 0.05 (quieter)
    quiet_noise = np.random.normal(0, 0.05, 1000).astype(np.float32)
    buffer.push(quiet_noise.tobytes())

    # Check that buffer is still enabled
    assert buffer.is_enabled()

    # Push 4000 samples of noise with amplitude 0.01 (even quieter)
    very_quiet_noise = np.random.normal(0, 0.01, 4000).astype(np.float32)
    buffer.push(very_quiet_noise.tobytes())

    # Check that buffer is disabled
    assert not buffer.is_enabled()

def test_reset():
    # Initialize a buffer with space for 10000 samples
    buffer = BufferWithQuiet(10000, 2000, dtype=np.float32, quiet_threshold=0.1)

    # Push 5000 samples of noise with amplitude 1.0
    noise = np.random.normal(0, 1.0, 5000).astype(np.float32)
    buffer.push(noise.tobytes())

    # Reset the buffer
    buffer.reset()

    # Check that the buffer is empty and enabled
    assert buffer.is_enabled()
    assert np.all(buffer.get() == 0)

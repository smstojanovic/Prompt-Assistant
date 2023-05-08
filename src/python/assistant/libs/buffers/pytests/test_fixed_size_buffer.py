import numpy as np
import pytest

from assistant.libs.buffers.fixed_size_buffer import FixedSizeBuffer

BUFFER_SIZE = 1024

@pytest.fixture
def buffer():
    buffer_size = BUFFER_SIZE*3
    return FixedSizeBuffer(buffer_size)


def test_fixed_size_buffer_push_and_get(buffer):
    # Generate some test data
    num_samples = BUFFER_SIZE*2
    test_data = np.random.randn(num_samples).astype(np.float32)

    # Push the test data to the buffer
    buffer.push(test_data)

    # Read back the test data from the buffer
    read_data = buffer.get(num_samples)

    # Check that the data matches
    assert np.allclose(test_data[:BUFFER_SIZE], read_data[:BUFFER_SIZE], rtol=1e-5, atol=1e-5)

def test_fixed_size_buffer_push_and_get_overflow(buffer):
    # Generate some test data
    num_samples = BUFFER_SIZE*2
    test_data = np.random.randn(num_samples).astype(np.float32)
    test_data_2 = np.random.randn(num_samples).astype(np.float32)
    # Push the test data to the buffer
    buffer.push(test_data)
    buffer.push(test_data_2)

    # Read back the test data from the buffer
    read_data = buffer.get(BUFFER_SIZE*3)

    # Check that the data matches
    assert np.allclose(test_data[BUFFER_SIZE:], read_data[:BUFFER_SIZE], rtol=1e-5, atol=1e-5)
    # cCheck the original dat isn't in.
    assert not np.isin(test_data[:BUFFER_SIZE], read_data).all()
    # Check entirety of new buffer data exists.
    assert np.allclose(test_data_2, read_data[BUFFER_SIZE:], rtol=1e-5, atol=1e-5)
    

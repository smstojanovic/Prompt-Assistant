import wave

def save_audio(buffer_samples, filename, sample_rate=48000):

    SAMPLE_RATE = sample_rate

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(buffer_samples)

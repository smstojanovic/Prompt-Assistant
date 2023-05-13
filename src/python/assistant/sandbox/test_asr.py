import whisper

model = whisper.load_model("base")
model = whisper.load_model("tiny.en")


result = model.transcribe("test_resampled.wav")
print(result["text"])

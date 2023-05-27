# Torchserve Implementation

This implements Torchserve for running the models in this project for inference.

## Compiling a Model

torch-model-archiver —-model-name “<model name>” —version 1.0 —-extra-files “./config.json”,”./utils.py” —handler “<path to handler>” —export-path model_store


torch-model-archiver --model-name "jenkins_prompt" --version "1.0" --extra-files "./config.json","./models/audio_processors/audio_transcription.py" --handler "./models/jenkins_prompt.py" --export-path model_store --force

torch-model-archiver --model-name "jenkins_listen" --version "1.0" --extra-files "./config.json","./models/audio_processors/audio_transcription.py" --handler "./models/jenkins_listen.py" --export-path model_store --force

torch-model-archiver --model-name "jenkins_speak" --version "1.0" --extra-files "./voice_embeddings.json","./models/audio_processors/text_synthesizer.py" --handler "./models/jenkins_speak.py" --export-path model_store --force

torch-model-archiver --model-name "jenkins_perceive" --version "1.0" --extra-files "./models/modifications/vad.py","./models/audio_processors/audio_vad.py" --handler "./models/jenkins_perceive.py" --export-path model_store --force

torchserve --start --ts-config torchserve.config --model-store "model_store"
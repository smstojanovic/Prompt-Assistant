# Torchserve Implementation

This implements Torchserve for running the models in this project for inference.

## Compiling a Model

torch-model-archiver —-model-name “<model name>” —version 1.0 —-extra-files “./config.json”,”./utils.py” —handler “<path to handler>” —export-path model_store


torch-model-archiver --model-name "jenkins_prompt" --version "1.0" --extra-files "./config.json" --handler "./models/jenkins_prompt.py" --export-path model_store


torchserve --start --ts-config torchserve.config --model-store "model_store" --models jenkins_prompt=jenkins_prompt.mar
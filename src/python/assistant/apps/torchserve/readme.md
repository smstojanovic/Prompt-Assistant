# Torchserve Implementation

This implements Torchserve for running the models in this project for inference.

## Compiling a Model

torch-model-archiver —-model-name “<model name>” —version 1.0 —-extra-files “./config.json”,”./utils.py” —handler “<path to handler>” —export-path model_store



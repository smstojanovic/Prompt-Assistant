from assistant.apps.torchserve.models.audio_processors.text_synthesizer import TextSynthesizeHandler
import json

class MockContext:
    system_properties = {'model_dir' : 'test'}
    manifest = None

with open('src/python/assistant/apps/torchserve/voice_embeddings.json','r') as f:
    voice_embedding = json.load(f)

ctx=  MockContext()
handler = TextSynthesizeHandler()

ctx.system_properties['voice_embedding'] = voice_embedding['voice_embedding']
handler.initialize(ctx)

input_data = [
	{'body' : {'data' : 'some data here'}},
]

data = handler.preprocess(input_data)
data = handler.inference(data)
data = handler.postprocess(data)

print(data)
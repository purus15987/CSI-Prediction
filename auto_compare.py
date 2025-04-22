import subprocess

channels = [128, 256, 512]
kmphs = [3, 30, 120]
models = ['stem', 'transformer', 'lstm']

for model in models:
    for channel in channels:
        for kmph in kmphs:
            print(f'\nRunning comparing model: {model} for channel={channel}, kmph={kmph}...\n')
            command = [
                'python', 'compare.py',
                '--model_name', str(model),
                '--encoded_dim', str(channel),
                '--kmph', str(kmph)
            ]
            subprocess.run(command)
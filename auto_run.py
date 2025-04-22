import subprocess

channels = [128, 256, 512]
kmphs = [3, 30, 120]
models = ['stem', 'transformer', 'lstm']

for model in models:
    for channel in channels:
        for kmph in kmphs:
            print(f'\nRunning training for model={model} channel={channel}, kmph={kmph}...\n')
            command = [
                'python', 'train.py',
                '--channel', str(channel),
                '--model_name', str(model),
                '--kmph', str(kmph)
            ]
            subprocess.run(command)
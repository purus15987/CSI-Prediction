import os
import torch
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from models.handler import train, test

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='CSI Prediction/X')
parser.add_argument('--model_name', type=str, default='stem')
parser.add_argument('--channel', type=str, default=128)
parser.add_argument('--kmph', type=int, default=30)
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)



args = parser.parse_args()
print(f'Training configs: {args}')
file_name = f'X_{args.channel}_{args.kmph}kmph'
# folders for output
result_train_file = os.path.join(f'{args.model_name}_output', file_name, 'train')
result_test_file = os.path.join(f'{args.model_name}_output', file_name, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
# load dataset file
data_file = os.path.join(args.dataset,file_name+'.pt')
data = torch.load(data_file)
data = data.numpy()
data = pd.DataFrame(data)
print("shape:",data.shape)
# split data
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]



torch.manual_seed(0)
if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(train_data, valid_data, args, result_train_file)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')
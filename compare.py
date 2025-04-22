import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import scipy.io as sio 
from scipy.io import savemat
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from collections import OrderedDict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--encoded_dim', type=int, default=128)
parser.add_argument('--model_name', type=str, default='stem')
parser.add_argument('--kmph', type=int, default=30)
parser.add_argument('--samples', type=int, default=10)
args = parser.parse_args()


img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
encoded_dim = args.encoded_dim #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
img_size = 32
in_chans = 2
num_heads = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
depth = 1
p = 0.
attn_p = 0.
qkv_bias=True
window = 8

class GroupAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super(GroupAttention, self).__init__()

        self.num_heads = num_heads
        head_dim = img_size // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(img_size, img_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(img_size, img_size)
        self.ws = window

    def forward(self, x):
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, C, h_group, self.ws, W)
        qkv = self.qkv(x).reshape(B, C, total_groups, -1, 3, self.num_heads, self.ws // self.num_heads).permute(4, 0, 1, 2, 5, 3, 6)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        return x

class GlobalAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super().__init__()

        self.dim = img_size
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim//window, self.dim//window * 2, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        self.sr = nn.Conv2d(2, 2, kernel_size=window, stride=window)
        self.norm = nn.LayerNorm(self.dim//window)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1, self.dim//window, self.dim//window).permute(0,1,3,2,4)
        x_ = self.sr(x).reshape(B, C, -1, self.dim//window, self.dim//window)
        x_ = self.norm(x_)
        kv = self.kv(x_).reshape(B, C, -1, 2, self.dim//window, self.dim//window).permute(3,0,1,4,2,5)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)

        return x

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.cc1 = nn.Linear(img_size, img_size)
        self.cc2 = nn.Linear(img_size, img_size)
        self.act = nn.GELU()

    def forward(self, x):

        x = self.cc1(x)
        x = self.act(x)
        x = self.cc2(x)

        return x


class WTL(nn.Module):
    def __init__(self, num_heads, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(img_size, eps=1e-6)
        self.attn1 = GroupAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.attn2 = GlobalAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm4 = nn.LayerNorm(img_size, eps=1e-6)
        self.mlp1 = MLP()
        self.mlp2 = MLP()

    def forward(self, x):

        x = x + self.attn1(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.mlp2(self.norm4(x))

        return x

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            img_size=img_size,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
    ):
        super().__init__()


        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.conv1 = nn.Conv2d(2,16, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(16,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(2*img_size*img_size, encoded_dim)

    def forward(self, x):

        n_samples = x.shape[0]
        x = self.conv1(x)
        x = self.conv5(x)
        X = x 

        for block in self.blocks:
            x = block(x)
        x = self.norm3(x)
        x = self.convT(x) 
        x = X + self.conv4(x)
        x = self.norm2(x)
        x = x.reshape(n_samples,2*img_size*img_size)
        x = self.fc(x)
        return x


class Decoder(nn.Module):   
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(encoded_dim, img_channels*img_size*img_size)
        self.act = nn.Sigmoid()
        self.conv5 = nn.Conv2d(2,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)

        self.dense_layers = nn.Sequential(
            nn.Linear(encoded_dim, img_total)
        )

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)

    def forward(self, x):
        img = self.dense_layers(x)
        img = img.view(-1, img_channels, img_height, img_width)

        out = self.decoder_feature(img)
        x = self.conv5(img)

        for block in self.blocks:
            x = block((x+out))

        x = self.norm2(x)
        x = self.convT(x)
        x = self.conv4(x) 

        for block in self.blocks:
            x = block((x+out))

        x = self.norm3(x)

        x = self.act(x) 

        return x

cr = encoded_dim
mode = 5
hori = 5
kmph = args.kmph
file_path = f'{args.model_name}_output/X_{cr}_{kmph}kmph/test'

x_test = torch.tensor(np.loadtxt(f"{file_path}/target.csv", delimiter=','), dtype=torch.float32)
x_hat  = torch.tensor(np.loadtxt(f"{file_path}/predict.csv", delimiter=','), dtype=torch.float32)

print(x_test.size())

Decoder_model = Decoder()
decoder = torch.load(f"decoder_{cr}.pth")
Decoder_model.load_state_dict(decoder)
Decoder_model.eval()
Decoder_model.to('cpu')

nmse = np.zeros((10,1))
corr = np.zeros((10,1))
se   = np.zeros((10,1))



with torch.no_grad():
    h_test = Decoder_model.forward(x_test)
    h_hat = Decoder_model.forward(x_hat)


# print(h_test.size())
h_test = h_test.to('cpu')
h_hat = h_hat.to('cpu')


# print(h_test.size())
h_test = h_test.numpy()
h_hat = h_hat.numpy()



h_test_real = np.reshape(h_test[:, 0, :, :], (len(h_test), -1))
h_test_imag = np.reshape(h_test[:, 1, :, :], (len(h_test), -1))
# h_test_C = h_test_real + 1j*(h_test_imag)
h_test_C = h_test_real-0.5 + 1j*(h_test_imag-0.5)
h_hat_real = np.reshape(h_hat[:, 0, :, :], (len(h_hat), -1))
h_hat_imag = np.reshape(h_hat[:, 1, :, :], (len(h_hat), -1))
# h_hat_C = h_hat_real + 1j*(h_hat_imag)
h_hat_C = h_hat_real-0.5 + 1j*(h_hat_imag-0.5)
power = np.sum(abs(h_test_C)**2, axis=1)
mse = np.sum(abs(h_test_C-h_hat_C)**2, axis=1)


H_test = np.reshape(h_test_C, (-1, 32, 32))
print(np.shape(H_test))
H_hat = np.reshape(h_hat_C, (-1, 32, 32))



torch.save(torch.tensor(h_test[:,0,:,:]+1j*h_test[:,1,:,:]), f"{args.model_name}_results/H_test_{cr}_{kmph}_{mode}_{hori}")
torch.save(torch.tensor(h_hat[:,0,:,:]+1j*h_hat[:,1,:,:]), f"{args.model_name}_results/H_hat_{cr}_{kmph}_{mode}_{hori}")


import matplotlib.pyplot as plt
import numpy as np

n = args.samples  # number of samples to plot
order = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23,
         9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]

plt.figure(figsize=(20, 6))
plt.suptitle(f"Prediction vs Ground Truth ({file_path})", fontsize=16)

for i in range(n):
    idx = 10 * i  # step through test data with stride
    # --- Target / Ground Truth ---
    ax = plt.subplot(2, n, i + 1)
    h_testplo = np.abs(h_test[idx, 0] + 1j * h_test[idx, 1])
    H = h_testplo[order, :]
    plt.imshow(H, aspect='auto')
    ax.set_title(f"Target {i}")
    ax.axis('off')

    # --- Prediction ---
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = np.abs(h_hat[idx, 0] + 1j * h_hat[idx, 1])
    Ht = decoded_imgsplo[order, :]
    plt.imshow(Ht, aspect='auto')
    ax.set_title(f"Pred {i}")
    ax.axis('off')

plt.tight_layout()
plt.savefig(f"{args.model_name}_results/imgs/{cr}_{kmph}_{mode}_{hori}.png", dpi=300, bbox_inches='tight')
# plt.show()




# print(np.shape(x_test))

n1 = abs(np.sqrt(np.sum(np.conj(H_test)*H_test, axis=1)))
# print(np.shape(n1))
n2 = abs(np.sqrt(np.sum(np.conj(H_hat)*H_hat, axis=1)))
aa = abs(np.sum(np.conj(H_hat)*H_test, axis=1))
rho2 = np.mean(aa/(n1*n2), axis=0)
# print("NMSE is ", 10*math.log10(np.mean(mse/power)))
# print("Correlation is ", np.max(rho2))
# print("SE is ", np.mean(capacity))


# print("NMSE is ", 10*math.log10(np.mean(mse1/power)))
# print("Correlation is ", np.mean(rho3))

nmse[0] = 10*math.log10(np.mean(mse/power))
corr[0] = np.max(rho2)

print("Total NMSE is ", np.mean(nmse))
print("Total Corr is ", np.mean(corr))

results = []
results.append([args.model_name, encoded_dim, kmph, np.mean(nmse)])

# Path where you want to save the CSV file
csv_filename = f"NMSE_results.csv"

# Check if the file exists to decide whether to write headers
file_exists = False
try:
    with open(csv_filename, 'r') as f:
        file_exists = True
except FileNotFoundError:
    pass

# Write the results to the CSV file
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers only if the file is new
    if not file_exists:
        writer.writerow(['Model', 'Encoded Dim', 'Speed (kmph)', 'NMSE'])

    # Write the actual result
    writer.writerows(results)

print(f"NMSE results saved to {csv_filename}")
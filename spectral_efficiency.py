import torch
import numpy as np
import matplotlib.pyplot as plt

# SNR values (as in your original code)
kmph = 30
ebn = list(range(0, 31, 5))
snr_list = [10 ** (x / 10) for x in ebn]

# List of cr values
cr_values = [128, 256, 512]

# Define the models
models = ['stem', 'transformer', 'LSTM']

# Placeholder for sum rates
sr = {model: [] for model in models}

many = 96

# Function to compute sum rate
def compute_sumrate(H_true, H_hat, snr):
    H_true = H_true.resolve_conj().cpu().detach().numpy()
    H_hat = H_hat.resolve_conj().cpu().detach().numpy()
    """Compute sum rate for one sample and one SNR value."""
    H_true = H_true / np.linalg.norm(H_true)
    H_hat = H_hat / np.linalg.norm(H_hat)

    power = np.zeros(32)
    for k in range(32):
        ht = H_hat[:, k].conj().T
        h = H_true[:, k]
        numerator = np.abs(np.dot(h, ht)) ** 2
        denominator = np.abs(np.dot(ht.T, ht)) ** 2
        power[k] = numerator / denominator if np.any(denominator != 0) else 0

    capacity = np.sum(np.log2(1 + (np.abs(power) ** 2) * snr / 32)) / 32
    return capacity

# Load the data and compute sum rates
for model in models:
    for cr in cr_values:
        # Load H_test and H_hat for the current cr and model
        H_test = torch.load(f'{model}_results/H_test_{cr}_{kmph}_5_5') # (batch, 32, 32)
        H_hat = torch.load(f'{model}_results/H_hat_{cr}_{kmph}_5_5') # (batch, 32, 32)
        # print(H_test.shape, H_hat.shape)

        # Calculate sum rate for each SNR value
        sr_capacity = []
        for snr_value in snr_list:
            capacities = []
            for i in range(many):
                capacity = compute_sumrate(H_test[i], H_hat[i], snr_value)
                capacities.append([capacity])
            sr_capacity.append(np.mean(capacities))
        sr[model].append(sr_capacity)


# Plotting the results
plt.figure(figsize=(10, 6))
for model in models:
    for idx, cr in enumerate(cr_values):
        sr_values = sr[model][idx]
        plt.plot(ebn, sr_values, label=f'{model}, cr={cr}, max_value={max(sr_values)}')

plt.xlabel(r'$E_{b}/N_{0} (dB)$')
plt.ylabel('Spectral Efficiency (bps/Hz)')
plt.title(f'Spectral Efficiency vs SNR for Different Models and CR Values - $\mu = {kmph} Kmph$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"Results/Spectral_Efficiency_{kmph}Kmph.png", dpi=300, bbox_inches='tight')
# plt.show()
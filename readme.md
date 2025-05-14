# CSI Prediction

This repository explores and compares deep learning models for **Channel State Information (CSI) prediction** in **massive MIMO systems**, with a focus on improving spectral and temporal predictions using various architectures like STEMGNN, Transformer, BiLSTM, and STNet.

> ✅ **Credits**: This work extends the contributions of [Sharan Mourya et al.](mailto:sharanmourya7@gmail.com) and is intended for comparative research and educational purposes.

For questions, collaborations, or feedback:

(a) Mailapalli Purushotham, Email: [purus15987@gmail.com](mailto:purus15987@gmail.com)

---

## 📌 Reference Projects

It builds upon and integrates prior work from the following repositories:

- [microsoft/StemGNN](https://github.com/microsoft/StemGNN)
    - Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia Zhu, Conguri Huang, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, Qi Zhang 
    - [Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting](https://arxiv.org/abs/2103.07719)
- [sharanmourya/CSI-Prediction](https://github.com/sharanmourya/CSI-Prediction) 
    - Sharan Mourya, Pavan Reddy, Sai Dhiraj Amuru
    - [Spectral Temporal Graph Neural Network for massive MIMO CSI Prediction.](https://ieeexplore.ieee.org/abstract/document/10457056)
- [sharanmourya/Pytorch_STNet](https://github.com/sharanmourya/Pytorch_STNet) 
    - Sharan Mourya, Sai Dhiraj Amuru
    - [A Spatially Separable Attention Mechanism For Massive MIMO CSI Feedback](https://arxiv.org/abs/2208.03369)

---

## 🚀 Project Overview

The goal of this project is to:
- Predict future CSI values in MIMO systems using deep learning models.
- Evaluate performance across varying channel dimensions and mobility scenarios.
- Use STNet for compression/decompression and integrate with STEMGNN pipeline.
- Models Implemented
    - `STEMGNN` – Graph neural network for time-series prediction.
    - `Transformer` – Sequence modeling architecture.
    - `BiLSTM` – Bidirectional recurrent neural network.
    - `STNet` – Attention-based CSI compression network (encoder-decoder).

---
## ⚙️ Requirements

- Core
    - Python >= 3.7
    - numpy >= 1.19.5
    - scipy >= 1.5.4
    - pandas >= 1.1.5
    - torch >= 1.9.0
    - torchvision
    - torchaudio
    - CUDA, install the appropriate torch version for your CUDA setup

- Plotting and visualization
    - matplotlib
    - seaborn

- Utility
    - tqdm
    - scikit-learn
    - tensorboard/wandb (optional)

Install dependencies with:
```bash
pip install -r requirements.txt
```

---
## 📂 Directory Structure

        CSI-Prediction/
        │
        ├── CSI-Prediction/X/                 # compressed channel matrices for Urban Macro (UMa) scenario
        │
        ├── data/                           # COST 2100 dataset
        │   ├── DATA_Htestin.mat
        │   └── ...
        │
        ├── assets/                          # Visualizations (NMSE, RMSE plots)
        │   ├── NMSE_comparison.png
        │   └── RMSE_comparison.png
        │
        ├── Results/                      # Stores output results and evaluation metrics
        ├── STNet Pretrained/             # Contains pretrained STNet models for CSI compression/decompression
        │
        ├── data_loader/                    # Data loading and preprocessing utilities
        │   └── data_utils.py
        │
        ├── models/                         # Deep learning model implementations
        │   ├── base_model.py               # STEMGNN architecture
        │   ├── transformer.py              # Transformer model for CSI prediction
        │   └── LSTM_model.py               # BiLSTM model for sequence modeling
        │   └── handler.py                  # BiLSTM model for sequence modeling
        │
        ├── utils/                          # Miscellaneous helper functions
        │   └── evaluation_metrics.py       # NMSE, RMSE, etc.
        │
        ├── Spectral_Temporal_Graph_Neural_Network_for_Massive_MIMO_CSI_Prediction.pdf  # Research paper
        ├── auto_compare.py               # Script for automated comparison of models
        ├── auto_run.py                   # Script to automate training and evaluation runs
        ├── compare.py                    # Script for comparing different model performances
        ├── practice.py                   # Script for practice or experimentation
        ├── requirements.txt              # Lists required Python packages
        ├── spectral_efficiency.py        # Calculates spectral efficiency metrics
        ├── stnet.py                      # Implementation of STNet for CSI compression
        └── train.py                      # Script to train the models


---
## 📊 Experimental Setup

### STNet 
- [A Spatially Separable Attention Mechanism For Massive MIMO CSI Feedback](https://arxiv.org/abs/2208.03369)

1) Dataset
    - For simulation purposes, we generate channel matrices from [COST2100 model](https://ieeexplore.ieee.org/document/6393523). Chao-Kai Wen and Shi Jin group provides a ready-made version of COST2100 dataset in [Dropbox](https://www.dropbox.com/scl/fo/tqhriijik2p76j7kfp9jl/h?rlkey=4r1zvjpv4lh5h4fpt7lbpus8c&dl=1).
    - After downloading, place it in the [data/]() directory
2) Training STNet
    - Firstly, choose the compression ratio 1/4, 1/8, 1/16, 1/32 or 1/64 by populating the variable encoded_dim with 512, 256, 128, 64 or 32 respectively.
    - Secondly, choose a scenario "indoor" or "outdoor" by assiging the variable envir the same.
    - Finally run the file STNet.py to begin training...

```bash
python stnet.py
```

    - pretrained weights of STNet encoder, decoder are saved in STNet [Pretrained/]() directory or you can use pretrained weights and use it in STEMGNN CSI-Prediction

### STEMGNN CSI-Prediction - 
1) Dataset
    -  We have provided the compressed channel matrices for Urban Macro (UMa) scenario with code word dimensions of 128, 256 and 512 is provided in the [google folder](https://drive.google.com/drive/folders/1mLpfNvQaA5PsV9R6soib4ptIIRMX5tQZ) shared here due to its large size.
    - Once dataset (train, test) is downloaded, we recommend to organize the folders according to project directory [CSI-Prediction/X/]()

2) Training and Evaluation
    - Set the window size, horizon, dataset name, and other training parameters in [train.py]() before running it.
    - you can use [auto_run.py]() to run all models with different scenarios and different channel dimensions.
    - After training is finished, the predicted channel matrices of train and test are stored in (train, test) [{args.model_name}_output/X_{args.channel}_{args.kmph}kmph/] automatically.

Train CSI predictors using:

```bash
python train.py
```

Or run batch jobs with:

```bash
python auto_run.py
```

3) STNet Decoder
    - Try to Reconstruct the original dataset (train, test) [[CSI-Prediction/X/]()] and predicted dataset (train, test) [{args.model_name}_output/X_{args.channel}_{args.kmph}kmph/] using [compare.py]()
    - you can use [auto_compare.py]() to reconstruct (train, test) for all models with different scenarios and different channel dimensions.

```bash
python compare.py
```

Run batch reconstructions with:

```bash
python auto_compare.py
```

4) Plotting Results
    - Run [spectral_eficiency.py]() by importing the decompressed channel matrices obtained in the previous step. This step produces the spectral efficiency plots of STEM GNN for various mobilities.
    - for comparative analysis - use models = ['stem', 'transformer', 'LSTM'] line no. 14 and cr_values = [128, 256, 512] line no. 11 
    - the results saved with predifined name at [Results/]()

```bash
python spectral_efficiency.py
```

Configure:

* `models = ['stem', 'transformer', 'LSTM']`
* `cr_values = [128, 256, 512]`

---

## 📈 Comparative Analysis

The repository includes [practice.py]() to evaluate model performance and visualize results:

- **Metrics**: Normalized Mean Square Error (NMSE), Root Mean Square Error (RMSE)
- **Plots**: `NMSE_comparison.png`, `RMSE_comparison.png`
---

---

## 📁 Full Results

Download the complete project output from:
📦 [Google Drive Folder](https://drive.google.com/drive/folders/1-IuNTeAtCa3ZIxTC7bv-wic9Sn3fMgJQ)

---

## 🤝 Contact

For questions, collaborations, or feedback:

📧 Mailapalli Purushotham

✉️ Email: [purus15987@gmail.com](mailto:purus15987@gmail.com)

🔗 [LinkedIn](https://www.linkedin.com/in/purushotham-mailapalli-0207471b3)

---

## 📄 License

This repository is available for **educational and non-commercial research** purposes only. For academic or commercial usage, please refer to the respective licenses of the base repositories.

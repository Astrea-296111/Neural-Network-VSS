

# MNIST-Neural-SS-Noise

本项目是一个基于深度学习的神经网络秘密共享（Neural Secret Sharing）模型实现。本项目以 MNIST 数据集为例，实现了 (k,n) 门限方案，通过引入多样性损失（Diversity Loss）和对抗性噪声诱导（Adversarial Noise Induction），确保在份额不足时重建结果为显式噪声，从而保障信息安全。




This project is an implementation of a Neural Secret Sharing model based on deep learning. Taking the MNIST dataset as an example, this project implements a (k, n) threshold scheme. By introducing Diversity Loss and Adversarial Noise Induction, it ensures that the reconstruction result is explicit noise when shares are insufficient, thereby guaranteeing information security.

## 目录 Table of Contents

- [项目结构 Project Structure](#项目结构-Project-Structure)
  - [项目文件 Project Files](#项目文件-Project-Files)
- [使用方法 Getting Started](#使用方法-Getting-Started)
  - [安装方法 Installation](#安装方法-Installation)
  - [代码示例 Core Example](#代码示例-Code-Example)
  - [训练步骤 Training Steps](#训练步骤-Training-Steps)
    - [阶段一：Warmup 建立重建能力 Phase 1: Warmup](#第一阶段-Phase-1)
    - [阶段二：Adversarial 显式噪声诱导 Phase 2: Adversarial](#第二阶段-Phase-2)
    - [阶段三：Hardening 门限收紧 Phase 3: Hardening](#第三阶段-Phase-3)
- [项目声明 Project Statement](#项目声明-Project-Statement)
- [许可证 License](#许可证-Lisense)

<h2 id="project">项目结构 Project Structure</h2>

<h3 id="project-file">项目文件 Project Files</h3>

├─ MINIST_Neural SS_noise_(5,5)-.ipynb (集成核心逻辑、训练与可视化的 Jupyter Notebook) 




 ├─ 0. 全局配置 (Global Configuration) 




 ├─ 1. 模型基础组件 (Basic Components: ResidualBlock) 




 ├─ 2. Encoder 架构 (Encoder: Secret Splitting) 




 ├─ 3. Decoder & Adversary 架构 (Decoder & Adversary) 




 ├─ 4. 训练逻辑与三阶段策略 (Training Logic) 




 ├─ 5. 测试与可视化 (Testing & Visualization) 

<h2 id="get-start">使用方法 Getting Started</h2>

<h3 id="install">安装方法 Installation</h3>

首先，克隆本项目并安装必要的 Python 库（建议使用 CUDA 加速）：




First, clone the project and install necessary Python libraries (CUDA acceleration recommended):

```
$ git clone https://github.com/Astrea-296111/Neural-Network-VSS
$ pip install torch torchvision matplotlib numpy tqdm

```

可通过全局配置模块自由选择参数k和n


```python
N_SHARES = 5           # n (total shares)
T_THRESH = 5           # t (threshold)
D_MODEL = 256          # d_model
width=8
D_SHARE = width*width        # d_share 大幅降低门限
IMG_SIZE = 28 * 28     # MNIST 展开维度

BATCH_SIZE = 128
EPOCHS_PHASE1 = 10     # Warmup
EPOCHS_PHASE2 = 30     # Adversarial
EPOCHS_PHASE3 = 10     # Hardening
```

<h3 id="example">代码示例 Code Example</h3>

本项目使用残差模块（ResidualBlock）构建 Encoder 和 Decoder。以下是项目中的核心模型定义示例：




This project uses ResidualBlock to build the Encoder and Decoder. The following is an example of the core model definition:

```python
# 模型组件示例：残差块 (ResidualBlock)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

```

<h3 id="Training">训练步骤 Training Steps</h3>

本项目采用三阶段课程学习策略，以平衡重建精度与安全性。




This project adopts a three-phase curriculum learning strategy to balance reconstruction accuracy and security.


<h4 id="phase_1">第一阶段 Phase 1</h4>

在第一阶段，模型学习如何将图像分割成 k 个份额并完整重建。




In the first phase, the model learns how to split an image into k shares and reconstruct it completely.

* **Loss**: `MSE(recon, original) + 0.1 * diversity_loss`
* **目的 (Aim)**: 确保所有份额组合后能恢复清晰原图。

<h4 id="phase_2">第二阶段 Phase 2</h4>

引入对抗网络（Adversary），迫使模型在份额少于门限值时输出类似均匀分布的噪声。




Introduce an Adversary to force the model to output noise-like images when the number of shares is less than the threshold.

```python
# 噪声诱导逻辑 (Noise Induction Logic)
# 当 k < T_THRESH 时，强制输出目标为随机噪声
noise_target = torch.rand_like(data).to(DEVICE)
adv_out = adversary(low_k_shares)
loss_adv = criterion_mse(adv_out, noise_target)

```

<h4 id="phase_3">第三阶段 Phase 3</h4>

进一步增强门限的严格性，使模型在t份share时仍无法获取任何有效信息。


```python
    for epoch in range(epochs):
        # Curriculum Schedule: 随 epoch 降低 k
        k_curr = max(T_THRESH, N_SHARES - int((N_SHARES - T_THRESH) * (epoch / (epochs * 0.5))))
        
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            
            opt_E.zero_grad()
            opt_D_recon.zero_grad()
            
            shares = encoder(data)
            
            # 随机挑选 k_curr 个 shares 给 Decoder
            indices = torch.randperm(N_SHARES)[:k_curr]
            shares_subset = shares[:, indices, :]
            
            reconstructed = decoder(shares_subset)
            l_recon = criterion_mse(reconstructed, data)
            l_div = diversity_loss(shares)
            
            loss = l_recon + gamma * l_div
            loss.backward()
            
            opt_E.step()
            opt_D_recon.step()
            total_loss += loss.item()

```


Further enhance the strictness of the threshold, making it impossible to obtain any valid information when $k$.


<h2 id="statement">项目声明 Project Statement</h2>

本项目的作者及单位：




The author and affiliation of this project:

```
项目名称（Project Name）：MINIST-Neural-SS-Noise
项目作者（Author）：Xiaotian Wu, Songyi Liao
作者单位（Affiliation）：暨南大学网络空间安全学院 (College of Cybersecurity, Jinan University)

```

若你使用本项目用于论文的实验，你可以引用本项目：




If you use this project for the experiment of the paper, you can cite this project:


    Author: Xiaotian Wu, Songyi Liao
    Project: [MINIST-Neural-SS-Noise](https://github.com/Astrea-296111/Neural-Network-VSS)
    
<h2 id="license">许可证 Lisense</h2>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目受 MIT 协议保护，详细内容请参阅 [LICENSE](https://github.com/Astrea-296111/Neural-Network-VSS/LICENSE).
Copyright (c) 2026 Astrea

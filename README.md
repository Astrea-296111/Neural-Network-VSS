# MNIST-Neural-SS-Noise

本项目是一个基于深度学习的神经网络秘密共享（Neural Secret Sharing）模型。本项目基于 PyTorch 框架，以 MNIST 数据集为例，不仅实现了图像的分割（编码）与重建（解码），还创新性地引入了多样性正则化（Diversity Loss）与对抗训练机制（Adversarial Noise Induction），通过三阶段训练策略确保在低于门限值时份额呈现显式随机噪声，从而保障信息安全性。




This project is a Neural Secret Sharing model based on deep learning. Based on PyTorch and taking the MNIST dataset as an example, this project not only implements image splitting (encoding) and reconstruction (decoding), but also innovatively introduces Diversity Loss and adversarial training mechanisms. Through a three-phase training strategy, it ensures that the shares exhibit explicit random noise below the threshold, thereby guaranteeing information security.

## 目录 Table of Contents

* [项目结构 Project Structure](https://www.google.com/search?q=%23%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84-project-structure)
* [核心文件 Core Files](https://www.google.com/search?q=%23%E6%A0%B8%E5%BF%83%E6%96%87%E4%BB%B6-core-files)
* [模块说明 Module Description](https://www.google.com/search?q=%23%E6%A8%A1%E5%9D%97%E8%AF%B4%E6%98%8E-module-description)


* [使用方法 Getting Started](https://www.google.com/search?q=%23%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95-getting-started)
* [安装方法 Installation](https://www.google.com/search?q=%23%E5%AE%89%E8%A3%85%E6%96%B9%E6%B3%95-installation)
* [训练步骤 Training Steps](https://www.google.com/search?q=%23%E8%AE%AD%E7%BB%83%E6%AD%A5%E9%AA%A4-training-steps)
* [阶段一：Warmup 建立重建能力 Phase 1: Warmup](https://www.google.com/search?q=%23%E9%98%B6%E6%AE%B5%E4%B8%80warmup-%E5%BB%BA%E7%AB%8B%E9%87%8D%E5%BB%BA%E8%83%BD%E5%8A%9B-phase-1-warmup)
* [阶段二：Adversarial 显式噪声诱导 Phase 2: Adversarial](https://www.google.com/search?q=%23%E9%98%B6%E6%AE%B5%E4%BA%8Cadversarial-%E6%98%BE%E5%BC%8F%E5%99%AA%E5%A3%B0%E8%AF%B1%E5%AF%BC-phase-2-adversarial)
* [阶段三：Hardening 门限收紧 Phase 3: Hardening](https://www.google.com/search?q=%23%E9%98%B6%E6%AE%B5%E4%B8%89hardening-%E9%97%A8%E9%99%90%E6%94%B6%E7%B4%A7-phase-3-hardening)


* [测试与可视化 Testing & Visualization](https://www.google.com/search?q=%23%E6%B5%8B%E8%AF%95%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96-testing--visualization)


* [项目声明 Project Statement](https://www.google.com/search?q=%23%E9%A1%B9%E7%9B%AE%E5%A3%B0%E6%98%8E-project-statement)
* [友情链接 Related Links](https://www.google.com/search?q=%23%E5%8F%8B%E6%83%85%E9%93%BE%E6%8E%A5-related-links)
* [许可证 License](https://www.google.com/search?q=%23%E8%AE%B8%E5%8F%AF%E8%AF%81-license)

由于本项目主要以交互式 Jupyter Notebook 的形式呈现，代码和训练逻辑被高度集成在单一文件中。




Since this project is mainly presented in the form of an interactive Jupyter Notebook, the code and training logic are highly integrated into a single file.

├─ MINIST_Neural SS_noise_(5,5)-.ipynb (核心代码与展示 Jupyter Notebook) 



该 Notebook 内部包含以下核心逻辑模块（Cells）：




The Notebook contains the following core logical modules (Cells):

 ├─ 0. 全局配置与设备设定 (Global configuration and device setting) 
 ├─ 1. 模型基础组件定义 (Basic model components: ResidualBlock) 
 ├─ 2. Encoder 架构 (Encoder architecture for secret splitting) 
 ├─ 3. Decoder & Adversary 架构 (Decoder/Adversary architecture for reconstruction) 
 ├─ 4. 训练逻辑与三阶段策略 (Training logic and three-phase strategy) 
 ├─ 5. 测试与可视化模块 (Testing and visualization module) 

首先，拉取本项目到本地，并确保您的环境中安装了 Jupyter 运行环境。




First, pull the project to the local machine, and ensure that the Jupyter environment is installed in your environment.

```
$ git clone <https://github.com/Astrea-296111/Neural-Network-VSS>
$ cd <Neural-Network-VSS/>

```

接着，安装本项目所需的依赖包。建议使用支持 CUDA 的 PyTorch 环境以加速训练。




Next, install the dependencies required for this project. It is recommended to use a PyTorch environment that supports CUDA to accelerate training.

```
$ pip install torch torchvision matplotlib numpy

```

最后，启动 Jupyter Notebook 并打开 `MINIST_Neural SS_noise_(5,5)-.ipynb` 文件即可直接运行。




Finally, start Jupyter Notebook and open the `MINIST_Neural SS_noise_(5,5)-.ipynb` file to run it directly.

本工具的核心在于 $(n, t) = (5, 5)$ 的秘密共享机制设定，其中 `N_SHARES = 5`，`T_THRESH = 5`。模型采用了创新的三阶段训练策略（Three-Phase Curriculum Strategy）：




The core of this tool lies in the $(n, t) = (5, 5)$ secret sharing mechanism setting, where `N_SHARES = 5` and `T_THRESH = 5`. The model adopts an innovative Three-Phase Curriculum Strategy:

在第一阶段，主要训练 Encoder 和 Decoder 的基础编解码能力。




In the first phase, the basic encoding and decoding capabilities of Encoder and Decoder are mainly trained.

* **目标**：使用所有的 $n$ 个 shares 重建原始图像。
* **损失函数**：重构损失（MSE Loss） + 多样性正则化（Diversity Loss, `-log det(Sh * Sh^T / n)`），用于确保生成的份额之间相互独立。

这是本项目最核心的对抗阶段，引入了独立的对抗网络（Adversary D）。




This is the core adversarial phase of the project, introducing an independent adversarial network (Adversary D).

* **逻辑**：Adversary 尝试从小于门限值 ($k < t$) 的份额中恢复原图。
* **惩罚机制**：Encoder 迫使 Adversary 输出均匀分布的随机噪声（`noise_target = torch.rand_like(data)`）。随着 Epoch 的增加，对抗损失权重 `lambda_adv` 将动态增加。

（注：代码中默认注释，可视需求开启）




(Note: Commented out by default in the code, can be turned on as needed)

* **目标**：使用 Curriculum Schedule，让可用于重建的份额数量 $k$ 随着训练进行，从 $n$ 逐步收缩到目标门限 $t$，进一步增强模型的鲁棒性。

执行第 5 个 Cell 中的 `test_and_visualize()` 函数。程序会自动提取测试集数据并生成 $(10 \times 11)$ 的对比网格图像。




Execute the `test_and_visualize()` function in the 5th cell. The program will automatically extract the test set data and generate a $(10 \times 11)$ comparison grid image.

* **展示内容**：包含原图 (Original)、5个独立份额 (Share 1-5)，以及使用不同数量份额（$k=1$ 到 $5$）的重建结果。
* **预期效果**：当 $k < 5$ 时，重建图像应显示为无法辨认的白噪声；当 $k = 5$ 时，完美重建原图。

本项目的作者及单位：




The author and affiliation of this project:

```
项目名称（Project Name）：MINIST-Neural-SS-Noise
项目作者（Author）：Xiaotian Wu, Songyi Liao
作者单位（Affiliation）：暨南大学网络空间安全学院 (College of Cybersecurity, Jinan University)

```

若你使用本项目用于论文的实验，你可以引用本项目：




If you use this project for the experiment of the paper, you can cite this project:

```
@misc{neuralss,
  author       = {Songyi, Liao},
  title        = {MINIST-Neural-SS-Noise: An Adversarial Neural Secret Sharing implementation},
  year         = {2026},
  howpublished = {\url{https://github.com/Astrea-296111/Neural-Network-VSS}}
}

```

1. [PyTorch Official Documentation](https://www.google.com/search?q=https://pytorch.org/docs/stable/index.html)

[MIT](https://www.google.com/search?q=LICENSE) © 2026 [廖松毅 Songyi Liao]

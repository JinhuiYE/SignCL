
# [SignCL: Improving Gloss-free Sign Language Translation by Reducing Representation Density](https://arxiv.org/abs/2405.14312)

## Overview

`SignCL` is a PyTorch module designed to facilitate the integration of contrastive learning into sign language translation models. It operates by sampling positive and negative pairs of frames from a sequence and computing a contrastive loss based on the distances between these pairs. This module can be integrated into both the pretraining and finetuning stages of a sign language translation model.

## Installation

To use `SignCL`, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision (optional, for dataset handling)
- Other standard libraries (e.g., random, numpy)

## Usage
Here's a step-by-step guide to integrating SignCL into your sign language translation model.
```bash
0. cl_criterion = SignCL()
1. frames_feature = model.encoder(src_input)
2. margin = min(20, max(10, int(num_frames // text_length * 2.3)))
3. cl_loss = cl_criterion(frames_feature, margin=margin)
4. total_loss = lambda_ * cl_loss + original_loss
```

### A. Usage example in your framework:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sign_cl import SignCL

# Define the Contrastive Loss Criterion
cl_criterion = SignCL(max_distance=32.0, pos_samples=2, neg_samples=4)

# Assume you have a model, data loader, and other necessary components
model = YourSignLanguageModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        src_input, text_input = batch['src'], batch['text']
        
        # Forward pass
        frames_feature = model.encoder(src_input)
        num_frames = frames_feature.size(1)
        text_length = len(text_input)  # Assuming text_input is the corresponding text
        margin = min(20, max(10, int(num_frames // text_length * 2.3))*2)
        
        cl_loss = cl_criterion(frames_feature, margin=margin)
        original_loss = ...  # Compute your original loss here
        lambda_ = 0.01  # Weight for the contrastive loss, adjust as necessary
        total_loss = lambda_ * cl_loss + original_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}")

```

### B. Usage example for GFSLT-VLP:
This example code was modified from [GFSLT-VLP GitHub](https://github.com/zhoubenjia/GFSLT-VLP). Please refer to their homepage to set up the environment and dataset.

To execute, use the following command:

```sh
bash examples/scripts.sh
```

This script will execute the training and evaluation process, demonstrating how to integrate the `SignCL` loss function into the GFSLT-VLP framework. We also included our self-reproduced `results` and `log.txt` on the CSL-Daily dataset (see [link](examples/GFSLT-VLP/out/0630_GF_SignCL)).

## Citation

Note if you find this code work for your research, please cite the following paper:

```bibtex
@inproceedings{ye2024improving,
  title={Improving Gloss-free Sign Language Translation by Reducing Representation Density},
  author={Ye, Jinhui and Wang, Xing and Jiao, Wenxiang and Liang, Junwei and Xiong, Hui},
  journal={arXiv preprint arXiv:2405.14312},
  year={2024}
}
```

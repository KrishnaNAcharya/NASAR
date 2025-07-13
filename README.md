# SAR Image Translation and Terrain Classification

## Project Overview
This repository contains advanced deep learning models for Synthetic Aperture Radar (SAR) to optical image translation, with a focus on terrain-aware processing. The project leverages state-of-the-art Generative Adversarial Networks (GANs) and terrain classification to enhance satellite imagery for remote sensing, environmental monitoring, and defense applications.

## Features
- **Terrain-Aware GANs**: Custom architectures with terrain-guided attention and adaptive normalization.
- **U-Net + PatchGAN**: High-fidelity image translation with skip connections and multi-scale discrimination.
- **Terrain Classification**: ResNet-based classifier for urban, grassland, agriculture, and barrenland types.
- **Mixed Precision & Multi-GPU Training**: Efficient, scalable training on modern hardware.
- **Comprehensive Metrics**: PSNR, SSIM, FID, IS, LPIPS, and more.
- **Automated Hyperparameter Tuning**: Keras Tuner integration for optimal model selection.

## Dataset
- **Source**: Sentinel-1 (SAR) and Sentinel-2 (optical) satellite images
- **Terrains**: `urban`, `grassland`, `agri`, `barrenland`
- **Structure**:
  - `Dataset/<terrain>/SAR/` — SAR images
  - `Dataset/<terrain>/Color/` — Corresponding optical images
- **Image Size**: 256x256 RGB

## Model Architectures
- **Generator**: U-Net backbone, terrain encoder, memory-efficient residual blocks, instance normalization, SiLU activation
- **Discriminator**: PatchGAN, spectral normalization, terrain-aware feature fusion, multi-scale outputs
- **Classifier**: ResNet-34 backbone, dropout regularization, one-hot terrain encoding

## Training & Evaluation
- **Distributed Training**: Multi-GPU, mixed precision, gradient accumulation
- **Losses**: Adversarial, L1, perceptual (EfficientNetB0/VGG16), gradient penalty, cycle consistency, feature matching
- **Metrics**: PSNR, SSIM, FID, IS, LPIPS
- **Visualization**: Side-by-side SAR, ground truth, and generated images

## Usage
### Requirements
- Python 3.8+
- PyTorch, torchvision, TensorFlow, Keras, scikit-image, matplotlib, numpy, tqdm, PIL, scipy

### Training
```bash
python SAR_Main_PyTorch.py
# or
python SAR_UNET_PATCHGAN.py
```

### Terrain Classifier
```bash
python SAR_Classification_training.py
python SAR_Classification_Testing.py
```

### Hyperparameter Tuning
```bash
python hyperparameter_tuning.py
```

### Resume Training
```bash
python resume_training.py
```

## Results

### Terrain Classification (ResNet34)
- **Accuracy:** 99.94%

### Image Translation (Conditional GAN)
**Key Evaluation Metrics:**
- **FID Score:** 108.18
- **SSIM:** 0.36
- **PSNR:** 19 dB
- **Inception Score (IS):** 3.07


## Applications
- Remote sensing, environmental monitoring, agricultural assessment, defense and security, all-weather satellite imaging

## Citation
If you use this code or models, please cite the repository and relevant papers.

---

**Contact**: Krishna N Acharya  
**Repository**: https://github.com/KrishnaNAcharya/SAR

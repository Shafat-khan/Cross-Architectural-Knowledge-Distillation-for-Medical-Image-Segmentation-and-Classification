# Cross-Architecture Knowledge Distillation for Brain Tumour Classification and Segmentation

A deep-learning research project exploring **cross-architecture knowledge distillation (CAKD)** and **multi-scale geometric feature fusion (MSGFF)** for brain tumour analysis using MRI images.

The framework combines convolutional neural networks, which capture local spatial details, with Vision Transformers, which model global context. Knowledge is then transferred to smaller student models designed to be more computationally efficient.

## Project at a Glance

- **Tasks:** Brain tumour classification and tumour-region segmentation
- **Classification teacher:** Hybrid VGG19 + Vision Transformer (ViT-B/16)
- **Classification student:** EfficientNet-B0
- **Segmentation teacher:** U-Net + Vision Transformer
- **Segmentation student:** Smaller U-Net
- **Dataset:** Figshare brain tumour MRI dataset
- **Main techniques:** Knowledge distillation, multi-scale feature matching, composite loss functions
- **Implementation environment:** Python and Google Colab; PyTorch is described in the report's implementation methodology

## Key Results

The results below are reported in the project report. They are validation results, not evidence of clinical readiness.

| Task | Model | Reported result |
|---|---|---:|
| Classification | EfficientNet-B0 distilled from hybrid VGG19 + ViT teacher | **93.31% validation accuracy** |
| Classification | Hybrid VGG19 + ViT teacher | 85.32% validation accuracy |
| Segmentation | Smaller U-Net student | **0.5943 Dice score** |
| Segmentation | U-Net + ViT teacher | 0.5749 Dice score |

The report therefore records a classification student validation accuracy of 93.31% and a segmentation student Dice score of 0.5943. Results depend on the dataset, preprocessing, split, and training configuration used in the experiments.

## Why This Approach?

CNNs are effective at capturing local patterns and spatial detail, while Vision Transformers can model long-range relationships across an image. The project combines these complementary representations in teacher models, then distils knowledge into smaller student networks.

The aim is to retain useful predictive information while reducing the computational burden associated with larger models. This is especially relevant when exploring medical-imaging systems for resource-constrained environments.

## Methodology

### 1. Dataset and preprocessing

The report uses the **Figshare brain tumour dataset**, described as containing 3,064 contrast-enhanced T1-weighted MRI images from 233 patients:

- Meningioma: 708 images
- Glioma: 1,426 images
- Pituitary tumour: 930 images

For classification, MATLAB files are loaded and image data are converted to three channels where required. Images are resized to **224 × 224**, converted to tensors, and normalised. Random horizontal flipping is used for training, while validation transformations omit random augmentation.

For segmentation, tumour masks are loaded where available or generated from tumour-border coordinates. Images and masks are resized to **256 × 256**; nearest-neighbour interpolation is used for masks to preserve their discrete labels. Image and mask augmentations are applied consistently.

### 2. Classification: hybrid teacher and student

The classification teacher combines two branches:

- **VGG19:** captures local visual and spatial features.
- **ViT-B/16:** captures global contextual information.

Their representations are concatenated and passed to a classifier. The student is **EfficientNet-B0**, trained using ground-truth labels, softened teacher outputs, and feature-level matching.

The report also compares dual-VGG19 and dual-ViT teacher variants.

### 3. Segmentation: U-Net + ViT teacher and smaller U-Net student

The segmentation teacher combines a U-Net encoder–decoder with a Vision Transformer branch. The projected ViT features are added element-wise to the U-Net bridge features, combining global context with local spatial information. The decoder then predicts a pixel-level tumour mask.

A smaller U-Net acts as the student, learning from both the ground-truth mask and the teacher.

### 4. Knowledge distillation and MSGFF

The framework uses composite losses to transfer both output behaviour and intermediate representations.

- **Classification:** cross-entropy for hard labels, KL-divergence for softened teacher/student outputs, and feature-level matching.
- **Segmentation:** binary cross-entropy with logits, output-level mean squared error (MSE), and multi-scale feature-level MSE.
- **Multi-scale geometric feature fusion (MSGFF):** aligns intermediate teacher and student features using projection and/or adaptive pooling before comparing them across feature scales.

This encourages the student to learn not only the teacher's predictions but also useful intermediate spatial representations.

## Architecture Overview

```text
CLASSIFICATION
MRI image
   ├── VGG19 branch ── local features ──┐
   └── ViT-B/16 branch ─ global context ├── concatenate ── classifier
                                        │
                         teacher outputs / intermediate features
                                        ↓
                                EfficientNet-B0 student
                                        ↓
                              Tumour class prediction


SEGMENTATION
MRI image
   ├── U-Net encoder ── bridge features ───────┐
   └── ViT branch ── projected global features ├── feature fusion
                                                ↓
                                      U-Net decoder + skips
                                                ↓
                                       Teacher tumour mask
                                                ↓
                                  Distillation + MSGFF losses
                                                ↓
                                        Smaller U-Net student
                                                ↓
                                       Predicted tumour mask
```

## Repository Contents

The existing repository screenshot shows notebooks, project reports, weekly reports, a problem statement, and presentation files. The exact notebook names include:

- `498R-2.ipynb`
- `CSE499B(main).ipynb`
- `segmentation(basic)ipynb-2.ipynb`

Other files in the repository include the final report, supporting PDFs, and presentations. Use the notebook that corresponds to the final experiment when documenting the exact training command or reproducing results.

## Tools and Technologies

- Python
- PyTorch
- VGG19
- Vision Transformer (ViT-B/16)
- EfficientNet-B0
- U-Net
- NumPy and image/data-processing utilities
- MATLAB-format dataset loading with `h5py`
- Google Colab and GPU acceleration

The exact package versions and a complete command-line training workflow are not specified here; check the notebooks before adding pinned dependencies or reproducibility instructions.

## Limitations and Future Work

The report notes computational constraints, the need for high-quality annotated medical datasets, and latency concerns associated with Vision Transformers. Further work should include:

- Evaluation on independent datasets and broader patient populations.
- More comprehensive reporting of Dice, IoU, sensitivity, specificity, and classification metrics.
- Repeated runs or cross-validation with clearly documented splits.
- Profiling model size, inference latency, and memory use on target hardware.
- Investigation of class imbalance and robustness to differences in MRI acquisition.
- Clinical validation before any diagnostic use.

## Responsible Use

This is an academic research prototype, **not a clinically validated diagnostic system**. Predictions should not be used to make medical decisions. Any real-world deployment would require rigorous independent validation, clinical oversight, privacy safeguards, and appropriate regulatory review.

## Academic Context

Senior Design Project, Department of Electrical and Computer Engineering, North South University, Spring 2025.

**Project title:** *Cross-Architecture Knowledge Distillation Framework with Multi-scale Geometric Feature Fusion in Medical Image Classification and Segmentation.*



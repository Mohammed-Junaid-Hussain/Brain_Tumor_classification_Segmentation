# Brain Tumor Classification and Segmentation using Hybrid Deep Learning Models

## Overview

This project involves developing a hybrid deep learning model for classifying and segmenting brain tumors in MRI images. The approach combines the strengths of two architectures: **Xception** for classification and **EfficientNetB7** for segmentation. The model achieved high accuracy in both tasks, demonstrating its utility in medical diagnostics.

## Project Goals

- **Classification:** Identify the presence of tumors in brain MRI images.
- **Segmentation:** Accurately segment the tumor regions within the MRI images.

## Model Architecture

- **Xception:** Used for the classification task, leveraging its powerful feature extraction capabilities.
- **EfficientNetB7:** Applied for the segmentation task, taking advantage of its efficiency and scalability in handling high-resolution images.

## Achievements

- **Classification Accuracy:** ~99.09%
- **Segmentation Accuracy:** ~98.09%
- **Grad-CAM:** Utilized to visualize the regions of interest (tumor areas) by generating heat maps, enhancing the interpretability of the segmentation results.

## Dataset

The model was trained and validated on a dataset of brain MRI images. Due to the lack of masked images, segmentation was a challenging task, but the model performed well, as evidenced by the achieved accuracy.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification-segmentation.git
   cd brain-tumor-classification-segmentation

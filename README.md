# FibonacciNet for Brain Tumor Classification

Implementation of the FibonacciNet architecture described in the paper: [FibonacciNet: Classification of Brain MRI Tumors Using Fibonacci Blocks](https://arxiv.org/html/2503.13928v1)

## Dataset
Brain Tumor MRI Images from Kaggle: [Dataset Link](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri/data)

## Model Architecture
The implementation follows the architecture described in the paper, with the following key components:
- 7 Fibonacci blocks
- Skip connections in blocks 2 -> 4 and 3 -> 5
- Adaptive learning rate schedule


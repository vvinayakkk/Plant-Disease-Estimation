# Plant Disease Detection System

This repository contains a comprehensive solution for detecting and classifying plant diseases using state-of-the-art deep learning models. The system leverages **VGG16**, **ResNet50**, and **InceptionV3** architectures to analyze and classify diseases in plants, achieving a remarkable **99% validation accuracy** with InceptionV3.


## Overview
Plant diseases significantly affect crop yield and food production. Early and accurate detection of diseases can help farmers take timely action, minimizing losses. This project applies deep learning to automate the classification of plant diseases based on leaf images. The system is trained on a custom dataset of labeled images, with advanced preprocessing and augmentation techniques to improve model robustness.

---

## Dataset
The dataset consists of high-quality images of plant leaves, categorized into various disease classes and healthy plants. Each image is labeled with the corresponding disease or "healthy" tag. Key characteristics:
- **Number of Classes**: 35 (including healthy)
- **Image Size**: Resized to 224x224 for model compatibility.
- **Augmentation**: Applied random rotations, flips, and cropping to enhance model generalization.

---

## Preprocessing
1. **Resizing**: All images were resized to 224x224 pixels to match the input dimensions of the models.
2. **Normalization**: Pixel values were normalized using ImageNet statistics (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`) for consistency with pre-trained weights.
3. **Data Augmentation**: Introduced variability using:
   - Random horizontal flips
   - Random rotations (up to 30°)
   - Random cropping

---

## Model Architectures

### VGG16
- **Architecture**: A deep network with 16 layers consisting of convolutional and fully connected layers. Known for its simplicity and uniform structure.
- **Advantages**:
  - Straightforward design
  - Pre-trained weights on ImageNet for transfer learning
- **Limitations**:
  - High computational cost
  - Larger number of parameters, increasing risk of overfitting

### ResNet50
- **Architecture**: A 50-layer deep residual network using skip connections to avoid vanishing gradients.
- **Advantages**:
  - Resolves vanishing gradient issues
  - Faster convergence
- **Limitations**:
  - Computationally expensive due to depth
  - Slightly lower performance compared to InceptionV3 for this task

### InceptionV3
- **Architecture**: Incorporates multiple filter sizes and efficient factorized convolutions within a single layer, reducing computational cost.
- **Advantages**:
  - High accuracy with fewer parameters
  - Efficient use of resources
  - Handles varying spatial features effectively
- **Limitations**:
  - Slightly complex design

---

## Performance Comparison
| Model      | Validation Accuracy | Precision | Recall | F1-Score |
|------------|----------------------|-----------|--------|----------|
| VGG16      | 92%                 | 0.90      | 0.91   | 0.91     |
| ResNet50   | 95%                 | 0.94      | 0.94   | 0.94     |
| InceptionV3| **99%**             | **0.99**  | **0.99**| **0.99** |

---

## Why InceptionV3 Performed Best
1. **Efficient Architecture**: 
   - InceptionV3 employs factorized convolutions and smaller kernels, reducing computation while capturing complex features effectively.
2. **Handling Spatial Variability**:
   - The model's ability to combine multiple filter sizes in a single layer allows it to detect both local and global features in leaf images.
3. **Reduced Overfitting**:
   - The reduced number of parameters (compared to VGG16) makes it less prone to overfitting, even on smaller datasets.
4. **Transfer Learning**:
   - Pre-trained weights on ImageNet enable the model to generalize better with minimal training time.
5. **Superior Feature Extraction**:
   - Handles fine-grained details in disease patterns, ensuring high precision and recall.

---

## Training Details
1. **Loss Function**: Cross-Entropy Loss for multi-class classification.
2. **Optimizer**: Adam optimizer with an adaptive learning rate for efficient convergence.
3. **Validation**: Early stopping based on validation loss to avoid overfitting.
4. **Hardware**: Training performed on an GPU P100  with 16GB memory on Kaggle.

---

## Results
- InceptionV3 achieved a **99% validation accuracy**, outperforming other models.
- High precision, recall, and F1-scores across all classes, with minimal misclassification.

---

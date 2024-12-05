# Brain Tumor Classification

This repository contains the implementation of three different models for brain tumor classification: a simple CNN model, an augmented CNN model, and a full-scale Spiking Neural Network (SNN) with temporal dynamics and spike-timing-dependent plasticity (STDP).

## Introduction

Brain tumor classification is a critical task in medical imaging, aiming to identify the type of tumor present in brain MRI images. This project compares the performance of three models on this task:
- **Simple Model**: A basic Convolutional Neural Network (CNN).
- **Augmented Model**: A CNN with data augmentation techniques.
- **SNN Model**: A Spiking Neural Network with temporal dynamics and STDP.

### Purpose

The primary goal of this project is to demonstrate that traditional CNN models, even with basic architectures, can outperform more complex SNN models on image classification tasks without specialized neuromorphic hardware. The simple model and augmented model produce significantly better results than the SNN, highlighting the current limitations of SNNs in practical applications.

## Dataset

This project uses the [Brain Tumor Dataset](https://www.kaggle.com/datasets/balamosay/brain-tumor-dataset). It consists of MRI images categorized into four classes:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

## Models

### Simple Model

The simple model is a basic CNN architecture. It achieved a validation accuracy of **89%** and a validation loss of **0.3701**.

#### Architecture

- **Input Layer**: Accepts images of size 180x180x3.
- **Rescaling Layer**: Normalizes pixel values to the range [0, 1].
- **Convolutional Layers**: 
  - Conv2D with 16 filters, kernel size 3x3, ReLU activation.
  - MaxPooling2D with pool size 2x2.
  - Conv2D with 32 filters, kernel size 3x3, ReLU activation.
  - MaxPooling2D with pool size 2x2.
- **Flatten Layer**: Flattens the 3D output to 1D.
- **Dense Layers**: 
  - Dense with 128 units, ReLU activation.
  - Dropout with rate 0.2.
  - Dense with 4 units, softmax activation.

### Augmented Model

The augmented model uses data augmentation techniques to improve generalization. It achieved a validation accuracy of **83%** and a validation loss of **0.427**.

#### Architecture

- **Input Layer**: Accepts images of size 180x180x3.
- **Rescaling Layer**: Normalizes pixel values to the range [0, 1].
- **Data Augmentation Layers**: 
  - RandomFlip for horizontal flipping.
  - RandomRotation with a factor of 0.1.
  - RandomZoom with a factor of 0.1.
- **Convolutional Layers**: 
  - Conv2D with 16 filters, kernel size 3x3, ReLU activation.
  - MaxPooling2D with pool size 2x2.
  - Dropout with rate 0.2.
  - Conv2D with 32 filters, kernel size 3x3, ReLU activation.
  - MaxPooling2D with pool size 2x2.
  - Dropout with rate 0.2.
- **Flatten Layer**: Flattens the 3D output to 1D.
- **Dense Layers**: 
  - Dense with 64 units, ReLU activation.
  - Dense with 4 units, softmax activation.

### SNN Model

The SNN model incorporates temporal dynamics and STDP. Despite its complexity, it did not perform as well as the CNN models without neuromorphic hardware. It achieved a validation accuracy of **29%** and a validation loss of **2023.996**.

#### Architecture

- **Input Layer**: Accepts images of size 128x128x3.
- **Rescaling Layer**: Normalizes pixel values to the range [0, 1].
- **Convolutional Layers**: 
  - Conv2D with 32 filters, kernel size 3x3, ReLU activation.
  - MaxPooling2D with pool size 2x2.
  - Conv2D with 64 filters, kernel size 3x3, ReLU activation.
  - MaxPooling2D with pool size 2x2.
  - Conv2D with 128 filters, kernel size 3x3, ReLU activation.
  - MaxPooling2D with pool size 2x2.
- **Flatten Layer**: Flattens the 3D output to 1D.
- **Custom SNN Layer (STDP)**: 
  - Simulates the behavior of biological neurons with temporal dynamics and spike-timing-dependent plasticity.
- **Dense Layers**: 
  - Dense with 64 units, ReLU activation.
  - Dropout with rate 0.5.
  - Dense with 4 units, softmax activation.

#### SNN Architecture Details

The SNN model was created from scratch, featuring a custom SNN layer that simulates the behavior of biological neurons. The architecture includes:
- **Temporal Dynamics**: The membrane potential of neurons is updated over time, incorporating a decay factor and input currents.
- **Spike Generation**: Neurons fire spikes when their membrane potential exceeds a certain threshold.
- **Spike-Timing-Dependent Plasticity (STDP)**: Synaptic weights are adjusted based on the timing of pre- and post-synaptic spikes, mimicking the learning mechanism observed in biological neurons.

Despite the sophisticated design, the SNN model faced challenges in achieving high accuracy, highlighting the current limitations of SNNs in practical applications without specialized hardware.

## Conclusion

This project demonstrates that traditional CNN models, even with basic architectures, can outperform more complex SNN models on image classification tasks without specialized neuromorphic hardware. The simple model achieved the highest validation accuracy, while the augmented model showed better alignment between training and validation metrics.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/brain-tumor-classification.git
    cd brain-tumor-classification
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the models**:
    - Simple Model: Open `models/simple_model/simple_model_training.ipynb` in Google Colab and run all cells.
    - Augmented Model: Open `models/augmented_model/augmented_model_training.ipynb` in Google Colab and run all cells.
    - SNN Model: Open `models/snn_model/snn_model_training.ipynb` in Google Colab and run all cells.

## Acknowledgements

This project was inspired by the need to explore different neural network architectures for medical image classification. Special thanks to the creators of the dataset and the open-source community for their invaluable resources.

## Contact

For any questions or suggestions, please contact marcosmasipcompany@gmail.com.


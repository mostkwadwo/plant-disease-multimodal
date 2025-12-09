# 🌿 Multimodal Plant Disease Detection System (RGB + Hyperspectral)

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-Container-2496ed?style=for-the-badge&logo=docker)
![Azure](https://img.shields.io/badge/Azure-Cloud-0078d4?style=for-the-badge&logo=microsoftazure)
![FastAPI](https://img.shields.io/badge/FastAPI-Live_Server-009688?style=for-the-badge&logo=fastapi)

> **🚀 Live API Demo:** [http://9.141.105.190/docs](http://9.141.105.190/docs)  
> *(Note: Hosted on Azure Linux VM. If the link is down, the VM is paused to save costs.)*

## 📋 Executive Summary
Traditional Computer Vision models often fail to detect plant stress until visible symptoms (yellowing, lesions) appear. By then, the crop yield is already compromised.

This project implements a **Multimodal Deep Learning Architecture** that fuses **Visual data (RGB)** with **Bio-Simulated Hyperspectral data**. By analyzing the "Red Edge" shift (700nm-1000nm) alongside visual texture, this model detects stress indicators with higher confidence than single-modality models.

---

## 🏗️ System Architecture

The system uses a **Late Fusion** strategy. Two separate neural networks process the inputs, and their learned features are concatenated to form a rich representation of the plant's health.

```mermaid
graph LR
    subgraph Inputs
    A[RGB Image <br/> (224x224)] --> |Spatial Feats| B(ResNet50 Encoder)
    C[Spectral Curve <br/> (400-1000nm)] --> |Frequency Feats| D(1D-CNN Encoder)
    end

    subgraph Fusion_Engine
    B --> E[Image Embedding <br/> (256 dim)]
    D --> F[Spectral Embedding <br/> (128 dim)]
    E --> G((Concatenate))
    F --> G
    G --> H[Fusion Layer + ReLU]
    end

    subgraph Output
    H --> I[Dropout 0.3]
    I --> J[Classifier Head]
    J --> K{Prediction: <br/> Healthy vs. Disease}
    end

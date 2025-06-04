# TSF-Net: Tempo-Spatial-Fusion Network for Deepfake Detection

This repository contains the official PyTorch implementation of **TSF-Net**, a hybrid deepfake detection framework that dynamically integrates spatial and temporal analysis. TSF-Net introduces several novel components for enhanced manipulation detection including Cross-Modal Attention Fusion (CMAF), Temporal Inconsistency Attention Module (TIAM), Artifact-Aware Loss Function (AALF), and Adaptive Computational Scaling (ACS).

### 📄 Manuscript
This code supports the following manuscript:

> **Tempo-Spatial-Fusion Network: A Novel Framework for Deepfake Detection through Dynamic Integration of Spatial and Temporal Features**  
> Asim Manzoor
> Submitted to PLOS Journal  
> Riphah International University, Islamabad, Pakistan

---

## 🧠 TSF-Net Highlights

- ✅ Dual-CNN feature extraction using EfficientNetV2L and modified XceptionNet
- ✅ Cross-modal attention fusion for dynamic integration of spatial features
- ✅ LSTM-based temporal attention for frame-to-frame inconsistency detection
- ✅ Custom artifact-aware loss function for robust training
- ✅ Adaptive computational scaling for resource-aware inference

---

## 🛠 Requirements

Install all required dependencies:

```bash
pip install -r requirements.txt

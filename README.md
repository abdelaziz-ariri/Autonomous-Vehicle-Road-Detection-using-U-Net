# 🚗 Autonomous Vehicle Road Detection using U-Net  
**Real-Time Semantic Segmentation for Driving Scenes**

---

## 📌 Project Overview

This project implements a **real-time semantic segmentation system** for **autonomous vehicle road detection** using the **U-Net deep learning architecture**.  
The model performs **pixel-wise classification** to identify key urban elements such as roads, sidewalks, vehicles, pedestrians, and infrastructure from driving videos.

The system is designed to be:
- **Accurate** (multi-class segmentation)
- **Efficient** (real-time video inference)
- **Modular** (easy to train, evaluate, and deploy)

---

## ✨ Key Features

- **U-Net Architecture**  
  Encoder–decoder structure with skip connections for precise spatial segmentation.

- **Real-Time Video Inference**  
  Optimized inference pipeline with GPU (CUDA) support.

- **Multi-View Visualization**
  - **Original View:** Raw input video
  - **Mask View:** Color-coded semantic segmentation
  - **Overlay View:** Mask blended with original frame for spatial alignment

- **Interactive Controls**
  - Pause / Resume playback
  - Save frames and masks
  - Instant exit via keyboard shortcuts

- **CamVid Dataset Support**  
  Built-in compatibility with the CamVid autonomous driving dataset.

---

## 📂 Project Structure
<img width="537" height="251" alt="image" src="https://github.com/user-attachments/assets/9ffe5964-d8a2-4522-83e5-8f1c66e317cc" />


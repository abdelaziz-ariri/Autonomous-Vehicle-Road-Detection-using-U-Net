# 🚗 Autonomous Vehicle Road Detection using U-Net  
**Real-Time Semantic Segmentation for Driving Scenes**

---

## 👥 Project Authors

This project was developed by:  
- **Abdelaziz Ariri**  
- **Othmane Hilal**

---

## 📌 Project Overview

The rapid advancement of autonomous driving systems requires accurate understanding of the surrounding environment. This project implements a **real-time semantic segmentation system** specifically designed for **road and urban scene detection** using the **U-Net deep learning architecture**.  

The system performs **pixel-level classification**, identifying essential elements in driving scenes, such as:
- Roads  
- Sidewalks  
- Vehicles  
- Pedestrians  
- Urban infrastructure (buildings, traffic signs, etc.)

By leveraging deep learning techniques, the system aims to improve **road perception** and **decision-making capabilities** for autonomous vehicles.

---

## ✨ Key Features

- **U-Net Architecture**  
  - Encoder–decoder structure with skip connections to preserve spatial information.  
  - Efficient design for real-time segmentation without compromising accuracy.

- **Real-Time Video Inference**  
  - Optimized pipeline for GPU acceleration (CUDA support).  
  - Capable of processing high-resolution driving videos with minimal latency.

- **Multi-View Visualization**
  - **Original View:** Displays the raw video feed.  
  - **Mask View:** Shows the semantic segmentation map with color-coded classes.  
  - **Overlay View:** Combines the original video and segmentation mask for spatial alignment.

- **Interactive Controls**
  - Pause / resume playback for frame-by-frame analysis.  
  - Save frames and masks for dataset augmentation or evaluation.  
  - Keyboard shortcuts for instant exit and control.

- **Dataset Support**
  - Built-in support for the **CamVid dataset**, widely used for autonomous driving research.  
  - Flexible pipeline to allow extension to other driving datasets.

- **Evaluation Metrics**
  - Supports standard segmentation metrics like **IoU (Intersection over Union)** and **pixel accuracy** for model performance assessment.

---

## 🛠️ Project Structure


<img width="537" height="251" alt="Project Structure" src="https://github.com/user-attachments/assets/9ffe5964-d8a2-4522-83e5-8f1c66e317cc" />



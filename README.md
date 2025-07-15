## 📄 Paper
# Deep learning-based estimation of time-series leaf area index and mean tilt angle using downward looking camera

This repository provides the **prediction code** for the *main algorithm* and *backup algorithm* proposed in our paper. 

## 📂 Repository Structure

- `main_algorithm/`: Code for the main deep learning prediction pipeline (CNN-GRU with 4-component segmentation).
- `backup_algorithm/`: Code for the backup prediction model used under cloudy conditions (binary CNN using vegetation + soil components).
- `Data/`：test data

## 📥 Download Pretrained Models & Test Data

You can download the pretrained `.pth` model files and example test datasets using the following links:

https://drive.google.com/drive/folders/14qIn4dS7FL1z5iKv_wAiz9CfQXuoC6Ah?usp=sharing

> 📝 Note: All `.pth` files are PyTorch models trained on field data collected in 2023, and the test set includes corresponding `.npy` image sequences and `.csv` ground truth files.

## 🚀 Getting Started

# SynMVCrowd
A large synthetic multi-view crowd counting and localization benchmark.

## Overview
SynMVCrowd: A Large Synthetic Benchmark for Multi-View Crowd Counting and Localization

Overview
This repository provides the code implementation for the baselines proposed in the paper "SynMVCrowd: A Large Synthetic Benchmark for Multi-view Crowd Counting and Localization". The SynMVCrowd benchmark is the largest synthetic dataset for multi-view and single-image crowd vision tasks, featuring 50 diverse scenes, 50 camera views per scene, 200 multi-view frames per scene, and crowd sizes ranging from 200 to 1000 people (average >500). It supports practical evaluation of multi-view crowd counting, localization, and domain transfer under cross-scene settings.
The dataset addresses limitations in existing multi-view datasets (e.g., Wildtrack, MultiviewX, CVCS) by offering larger scenes, higher crowd densities, variable weather conditions (clear, cloudy, rainy, etc.), light variations (0-24 hours), and diverse urban environments (parks, beaches, streets, etc.). It can also serve as a challenging benchmark for single-image crowd counting and localization.
Key contributions from the paper:

- Largest synthetic multi-view crowd dataset with 500,000 images (1920x1080 resolution).
- Strong baselines outperforming SOTA methods (e.g., MVDet, MVDeTr, SHOT, 3DROM) on cross-scene evaluation.
- Improved domain transfer for multi-view and single-image counting to real-world scenes.

The code includes training and testing scripts for multi-view detectors like MVDeTr, 
with support for datasets such as Wildtrack and  SynMVCrowd. Pretrained models from SynMVCrowd are loaded for enhanced performance.

## Data Statistics
| Dataset          | Type | Scenes | Size (m)   | Cameras   | Frames | Avg. Resolution | Total Counts | Min | Avg | Max  |
|------------------|------|--------|------------|-----------|--------|-----------------|-------------|-----|-----|------|
| PETS2009 [9]     | Real | 1      | -          | 3         | 1,899  | 576x768         | -           | 20  | -   | 40   |
| DukeMTMC [26]    | Real | 1      | -          | 4         | 989    | 1080x1920       | -           | 10  | -   | 30   |
| Wildtrack [8]    | Real | 1      | 12x36      | 7         | 400    | 1080x1920       | -           | -   | 20  | -    |
| MultiviewX [6]   | Syn  | 1      | 16x25      | 6         | 400    | 1080x1920       | -           | -   | 40  | -    |
| CityStreet [21]  | Real | 1      | 58x72      | 3         | 500    | 1520x2704       | 64K         | 70  | 128 | 150  |
| CVCS [5]         | Syn  | 31     | 90x80      | 60-120    | 3,100  | 1080x1920       | 418.5K      | 90  | 135 | 180  |
| **SynMVCrowd**   | Syn  | 50     | 100x120    | 50        | 10,000 | 1080x1920       | 5.3M        | 151 | 530 | 1,000|

- **Split**: Training (30 scenes), Validation (10 scenes), Test (10 scenes) in 3:1:1 ratio.
- **Annotations**: Head coordinates in image/world space, semantic segmentation, camera parameters, weather/time metadata.
- **Download**: Please click this url for downloading the SynMVCrowd datasets. https://www.dropbox.com/home/SynMVCrowd/SynMVCrowdDataset


## Installation
**Requirements**
``` 
conda create -n synmvc python=3.10 -y
pip install -r requirements.txt
```

## Setup

Clone the repository:
``` 
git clone https://github.com/zqyq/SynMVCrowd
cd SynMVCrowd/scripts
python main.py
```

Install dependencies:numpy, torchvision, tqdm, opencv-python
Download datasets (e.g., Wildtrack) and place in /mnt/d/Datasets/ (or update paths in code).

## Usage
The main script (main.py) handles training and testing of the MVDeTr model. It supports debug mode, deterministic training, and multi-GPU.
Command-Line Arguments

- dataset: Dataset to use (e.g., wildtrack, SynMVCrowd).
- arch: Backbone architecture (e.g., resnet18).
- num_cam: Number of cameras (default: 5).
- loss_type: Loss function (e.g., mse, ot, bce).
- epochs: Training epochs (default: 200).
- lr: Learning rate (default: 1e-5).
- pretrained_dir: Path to pretrained model (e.g., SynMVCrowd checkpoint).
- visualize: Enable visualization.
- test: Run in test mode only.

## Ackonwledgement
This work was supported in parts by National Natural Science Foundation of China (No. 62202312, U21B2023, U2001206), Guangdong Basic
and Applied Basic Research Foundation (2023B1515120026), DEGP Innovation
Team (2022KCXTD025), Shenzhen Science and Technology Program (KQTD
20210811090044003, RCJC20200714114435012), Guangdong Laboratory of Artificial
Intelligence and Digital Economy (SZ), and Scientific Development Funds from
Shenzhen University

## Reference
```
@article{Zhang_2026_SynMVCrowd,
    author  = {Q. Zhang and D. Chen and Y. Gong and H. Huang},
    title   = {{SynMVCrowd}: A Large Synthetic Benchmark for Multi-view Crowd Counting and Localization},
    journal = {International Journal of Computer Vision},
    year    = {2026},
    volume  = {134},
    pages   = {191},
    month   = mar,
    doi     = {10.1007/s11263-026-02778-3},
    url     = {https://doi.org/10.1007/s11263-026-02778-3}
}
```
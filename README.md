## [ICLR 2024 Spotlight] Bounding Box Stability against Feature Dropout Reflects Detector Generalization across Environments

#### 🚀🚀🚀 A brand-new problem of estimating the detector performance in an unlabeled test domain.

## Abstract

Bounding boxes uniquely characterize object detection, where a good detector gives accurate bounding boxes of categories of interest. However, in the real-world where test ground truths are not provided, it is non-trivial to find out whether bounding boxes are accurate, thus preventing us from assessing the detector generalization ability. In this work, we find under feature map dropout, good detectors tend to output bounding boxes whose locations do not change much, while bounding boxes of poor detectors will undergo noticeable position changes. We compute the box stability score (BoS score) to reflect this stability. Specifically, given an image, we compute a normal set of bounding boxes and a second set after feature map dropout. To obtain BoS score, we use bipartite matching to find the corresponding boxes between the two sets and compute the average Intersection over Union (IoU) across the entire test set. We contribute to finding that BoS score has a strong, positive correlation with detection accuracy measured by mean average precision (mAP) under various test environments. This relationship allows us to predict the accuracy of detectors on various real-world test sets without accessing test ground truths, verified on canonical detection tasks such as vehicle detection.

## PyTorch Implementation

This repository contains:

- the PyTorch implementation of BoS.
- the progress to construct a meta set for AutoEval in object detection.
- correlation study

Please follow the instruction below to install it.

### Prerequisites

- Linux (tested on Ubuntu 16.04LTS)
- NVIDIA GPU + CUDA CuDNN (tested on GTX 2080 Ti and A100)

**Dataset.** For the convenience of users, we have standardized the formats of the 10 existing object detection datasets, including COCO, BDD, Cityscapes, DETRAC, Exdark, Kitti, Self-driving, Roboflow, Udacity, and Traffic. For each domain, we have randomly selected 250 images containing vehicles.  These images are available for download via the following [link](https://drive.google.com/file/d/1bs1y04q_0VeSDTnex0i94gzK8vGXdx5r/view?usp=sharing). Please place the images under "PROJECT_DIR/data" 

**Model to be evaluated.** Users can download the model via the provided [link](https://drive.google.com/drive/folders/1zAFcSgl1vfzg0BUnyJqg8t_9FkZIXA5h?usp=sharing) and place it in the "PROJECT_DIR/work_dir" folder for convenience. Or users could train vehicle detection models on themselves and evaluate them. 

## Getting started
0. Install dependencies 
    ```bash
    # Energy-based AutoEval
    conda env create --name autoeval --file environment.yaml
    conda activate autoeval
    pip install -v -e .
    ```
1. Creat synthetic sets
    ```bash
    
    # By default it creates 50 synthetic sets for each domain
     bash scripts/metaset_generate/all.sh
    ```
    
2. Load detector and begin testing
    ```bash
    # Save test results under "PROJECT_DIR/res"
    bash scripts/autoeval/bos/all.sh
    ```

3. Correlation study
    ```bash
    # You will see "PROJECT_DIR/figs/correlation_mAP_bos.pdf"
    python analyze_correlation.py
    ```

## License
MIT

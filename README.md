# <img src="./assets/iggt_logo.png" alt="logo" width="30"/> IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction

This is the official repository for the paper:
> **IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction** 
>
> [Hao Li*](https://lifuguan.github.io/), [Zhengyu Zou*](), [Fangfu Liu*](https://scholar.google.com/citations?user=b-4FUVsAAAAJ&hl=zh-CN), [Xuanyang Zhang](https://scholar.google.com/citations?user=oPV20eMAAAAJ&hl=zh-CN), [Fangzhou Hong](https://scholar.google.com/citations?user=mhaiL5MAAAAJ&hl=zh-CN&oi=ao), [Yukang Cao](https://scholar.google.com/citations?user=1rIzYQgAAAAJ&hl=zh-CN&oi=ao), [Yushi Lan](https://scholar.google.com/citations?user=dTNZCUcAAAAJ&hl=zh-CN&oi=ao), [Manyuan Zhang](https://manyuan97.github.io/), [Gang Yu](https://www.skicyyu.org/), [Dingwen Zhang](https://teacher.nwpu.edu.cn/zdw2006yyy)<sup>†</sup>, and [Ziwei Liu](https://liuziwei7.github.io/)
>
> <sup>*</sup>Equal Contribution, <sup>†</sup>Project Leader, <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2709.svg" alt="email" width="16"/>Corresponding author.
>
> ### <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/arxiv.svg" alt="arXiv" width="20"/> [Paper](https://arxiv.org/abs/2510.22706) &nbsp; | &nbsp; <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/internetarchive.svg" alt="Website" width="20"/> [Website](https://lifuguan.github.io/IGGT_official) &nbsp; | &nbsp; <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" width="20"/> [Data](https://huggingface.co/datasets/lifuguan/InsScene-15K)  &nbsp; | &nbsp; <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" width="20"/> [Benchmark](https://huggingface.co/datasets/lifuguan/IGGT_Benchmark) 


<div align="center">
  <img src="./assets/demo_video.gif" alt="IGGT Demo" width="800"/>
</div>

## 🔍 Overview
IGGT introduces a novel transformer-based architecture for semantic 3D reconstruction that grounds instance-level understanding in geometric representations. Our method achieves state-of-the-art performance on multiple benchmarks while maintaining computational efficiency.

**Key Features:**
- 🎯 Instance-grounded 3D feature learning
- 🏗️ Geometry-aware transformer architecture
- 📊 State-of-the-art performance on ScanNet and InsScene-15K
- ⚡ Efficient inference with multi-view consistency

## 📝 To-Do List

- [x] Release project paper
- [x] Release Benchmark (Segmentation, Track)
- [x] Release InsScene-15K dataset
- [] Release codebase
  - [x] Release model code
  - [] Release downstream task scripts
- [] Release pretrained models


## 🚀 Quick Start

## Installation

To set up the environment for this project, please follow these steps:

1. Create a new Conda environment with Python 3.10.0:
   ```bash
   conda create -n iggt python=3.10.0
   conda activate iggt
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: To accelerate clustering (DBSCAN) significantly, we highly recommend installing `cuml` from RAPIDS. Please refer to the [official installation guide](https://docs.rapids.ai/install/#selector) to choose the appropriate version for your system.


### Running the Demo

We provide `demo.py` to demonstrate IGGT's capabilities in 3D scene reconstruction and segmentation.

#### 1. Data Organization

We provide sample scenes in the `iggt_demo` directory (e.g., `iggt_demo/demo1` to `iggt_demo/demo9`).
For your own data, please organize it with the following structure:

```text
scene_name/
└── images/           # Input images (sorted by filename)
    ├── 00000.jpg
    ├── 00001.jpg
    └── ...
```

(Optional) For evaluation against ground truth:
```text
scene_name/
├── depth/            # Ground truth depth maps
└── cam/              # Camera parameters (.npz files)
```

#### 2. Usage

Configure the paths in `demo.py`:
- `MODEL_PATH`: Path to the pretrained checkpoint.
- `TARGET_DIR`: Path to your input data directory.
- `SAVE_DIR`: Path where results will be saved.

You can also adjust the `CLUSTERING_CONFIG` in `demo.py` to optimize segmentation results:
- `eps`: DBSCAN epsilon parameter (default: 0.01). Controls the maximum distance between points to be considered neighbors.
- `min_samples`: Minimum samples for a core point (default: 100).
- `min_cluster_size`: Minimum size for a valid cluster (default: 500).
- `knn_k`: Number of neighbors for spatial smoothing (default: 20).

Then run the script:
```bash
python demo.py
```

The script will generate:
- **3D Visualizations**: `.glb` files for RGB, Mask, and PCA features.
- **Depth Maps**: Visualizations with various colormaps in `pred_depths/`.
- **Segmentation**: DBSCAN and PCA masks in `dbscan_masks/` and `colored_pca/`.

<p align="center">
  <img src="./assets/iggt_demo.png" alt="IGGT Demo Example" width="80%">
</p>

**Figure:** Example 3D scene segmentation and reconstruction by IGGT.




## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{li2025iggt,
  title={IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction},
  author={Li, Hao and Zou, Zhengyu and Liu, Fangfu and Zhang, Xuanyang and Hong, Fangzhou and Cao, Yukang and Lan, Yushi and Zhang, Manyuan and Yu, Gang and Zhang, Dingwen and others},
  journal={arXiv preprint arXiv:2510.22706},
  year={2025}
}
```

## 📄 License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

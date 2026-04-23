<!-- PROJECT LOGO -->
<h1 align="center">Anonymous github</h1>

<p align="center">
This is not an officially endorsed Google product.
</p>

<p align="center">
    <img src="./media/teaser.png" alt="teaser_image" width="100%">
</p>

<p align="center">
<strong>Anonymous github</strong>
</p>

<p align="center">
    <img src="./media/framework.png" alt="framework" width="100%">
</p>
<p align="center">
<strong>Anonymous Architecture</strong>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#data-download">Data Download</a></li>
    <li><a href="#run">Run</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Installation

1. Clone the repo using the `--recursive` flag.

```bash
git clone --recursive https://github.com/Brave2003/Try-in-Splat-SLAM.git
mv Try-in-Splat-SLAM Anonymous   # 可选：保持项目目录名为 Anonymous
cd Anonymous
```

2. Create a new conda environment (recommended name: `splat`).

```bash
conda create --name splat python=3.10
conda activate splat
```

3. Install PyTorch (CUDA 11.7 example).

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -c "import torch; print(torch.cuda.is_available())"
```

4. Update depth rendering hyperparameter in third-party library.

By default, the gaussian rasterizer does not render gaussians that are closer than 0.2m in front of the camera. In monocular setting this can hurt rendering. Change the threshold to 0.001 at [auxiliary.h#L154](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose/blob/43e21bff91cd24986ee3dd52fe0bb06952e50ec7/cuda_rasterizer/auxiliary.h#L154):

```cpp
if (p_view.z <= 0.001f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
```

5. (Optional, highly recommended) Install SAM2 in a separate conda env (`sam2env`) for dynamic mask refinement.

```bash
conda deactivate
conda create -n sam2env python=3.10
conda activate sam2env

# Example from pytorch.org (please choose your platform/CUDA there)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

cd thirdparty/sam2
pip install -e ".[notebooks]"
# CPU-only build (optional):
# SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
cd ../../
```

`utils/sam2_bridge.py` searches SAM2 runtime in this order:

- `SAM2_PYTHON` (direct python executable)
- `SAM2_ENV_PREFIX`
- `SAM2_ENV_NAME` + `CONDA_ROOT`
- fallback: `~/miniconda3/envs/sam2env/bin/python`

Example:

```bash
export CONDA_ROOT=/data/opt/miniconda3
export SAM2_ENV_NAME=sam2env
# or directly:
# export SAM2_PYTHON=/data/opt/miniconda3/envs/sam2env/bin/python
```

6. Install third-party dependencies.

提示:
- 如遇到 build isolation 下找不到 torch，可加 `--no-build-isolation`。
- `simple-knn` 编译问题可参考下方备注。

```bash
python -m pip install -e thirdparty/lietorch/
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/ --no-build-isolation
python -m pip install -e thirdparty/simple-knn/ --no-build-isolation
python -m pip install -e thirdparty/evaluate_3d_reconstruction_lib/
```

`simple-knn` 编译不过时可尝试（仅供参考）：

```bash
# 新建文件 thirdparty/simple-knn/simple_knn/__init__.py
from . import _C
```

并在 `thirdparty/simple-knn/setup.py` 中确保包含:

```python
name="simple_knn",
packages=["simple_knn"],
```

7. Check installation.

```bash
python -c "import torch; import lietorch; import simple_knn; import diff_gaussian_rasterization; print(torch.cuda.is_available())"
```

8. Install droid backends and remaining requirements.

```bash
python -m pip install -e .
pip install "torch-scatter==2.0.9" --no-build-isolation
pip install git+https://github.com/KinglittleQ/torch-batch-svd.git --no-build-isolation
python -m pip install -r requirements.txt
python -m pip install pytorch-lightning==1.9 --no-deps
pip install lightning-utilities==0.4.2
pip install ultralytics
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```

9. Download pretrained models (store all in `pretrained/`).

本仓库默认脚本是：

```bash
bash scripts/download_pretrained_model.sh
```

该脚本下载的是旧版预训练包（包含 `omnidata_dpt_depth_v2.ckpt`）。

若你使用当前分支中的动态掩码/法向/光流流程，请确保以下权重在 `pretrained/` 下：

- `droid.pth`
- `depth_anything_v2_vitl.pth`
- `yolo11l-seg.pt`
- `sam2.1_hiera_base_plus.pt`
- `raft-things.pth`
- `dsine.pt`（参考 `thirdparty/DSINE/README.md`）

推荐目录结构：

```bash
pretrained/
├── droid.pth
├── depth_anything_v2_vitl.pth
├── yolo11l-seg.pt
├── sam2.1_hiera_base_plus.pt
├── raft-things.pth
└── dsine.pt
```

## Data Download

### Replica

```bash
bash scripts/download_replica.sh
bash scripts/download_replica_cull_mesh.sh
```

### TUM-RGBD

```bash
bash scripts/download_tum.sh
```

请在场景配置文件里把 `input_folder` 改成你的本地数据路径。

### ScanNet

请按 [ScanNet 官网](http://www.scan-net.org/) 下载，并使用官方 `SensReader` 脚本提取彩色/深度帧。

## Run

For running Anonymous, each scene has a config, where `input_folder` and `output` should be set correctly.

### Replica

```bash
python run.py configs/Replica/office0.yaml
```

### TUM-RGBD

```bash
python run.py configs/TUM_RGBD/TUM/sitrpy.yaml
```

### BONN

```bash
python run.py configs/TUM_RGBD/BONN/sync2.yaml
```

### ScanNet

```bash
python run.py configs/Scannet/scene0000.yaml
```

After reconstruction, trajectory evaluation runs automatically (and some configs also run geometry/render metrics).

## Run tracking without mapping

Anonymous uses two processes (tracking and mapping). You can run tracking only via `--only_tracking`:

```bash
python run.py configs/Replica/office0.yaml --only_tracking
python run.py configs/TUM_RGBD/TUM/sitrpy.yaml --only_tracking
python run.py configs/TUM_RGBD/BONN/sync2.yaml --only_tracking
python run.py configs/Scannet/scene0000.yaml --only_tracking
```

## Acknowledgement

Our codebase is partially based on [splat-SLAM](https://github.com/google-research/Splat-SLAM), [GO-SLAM](https://github.com/youmi-zym/GO-SLAM), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [MonoGS](https://github.com/muskie82/MonoGS). We thank the authors for making these codebases publicly available.

## Reproducibility

There may be minor differences between the released codebase and the results reported in the paper. GPU hardware may also influence the final results, even with the same seed and conda environment.

## Citation

If you find this project useful, please cite the upstream Splat-SLAM paper:

```bibtex
@article{sandstrom2024splat,
  title={Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians},
  author={Sandstr{\"o}m, Erik and Tateno, Keisuke and Oechsle, Michael and Niemeyer, Michael and Van Gool, Luc and Oswald, Martin R and Tombari, Federico},
  journal={arXiv preprint arXiv:2405.16544},
  year={2024}
}
```

## Contact

For questions, please open an issue in this repository.

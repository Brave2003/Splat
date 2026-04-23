import os
import subprocess
import uuid

import numpy as np
import imageio.v2 as imageio  # 可换成 cv2.imwrite

# ---------------------------------------------------------
# 路径配置（相对 try 工程根目录）
# ---------------------------------------------------------

# 本文件所在目录：try/utils
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录：try
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
# SAM2 目录：try/thirdparty/sam2
SAM2_DIR = os.path.join(PROJECT_ROOT, "thirdparty", "sam2")

# run_sam2_once.py 的绝对路径
SAM2_RUNNER_PY = os.path.join(SAM2_DIR, "run_sam2_once.py")

# 默认的 conda 根目录与环境名（相对当前用户的 $HOME）
DEFAULT_CONDA_ROOT = os.path.expanduser(os.environ.get("CONDA_ROOT", "~/miniconda3"))
DEFAULT_SAM2_ENV_NAME = os.environ.get("SAM2_ENV_NAME", "sam2env")

# SAM2 环境前缀与 Python 路径（可通过环境变量覆盖）
SAM2_ENV_PREFIX = os.environ.get(
    "SAM2_ENV_PREFIX",
    os.path.join(DEFAULT_CONDA_ROOT, "envs", DEFAULT_SAM2_ENV_NAME),
)
SAM2_PYTHON = os.environ.get(
    "SAM2_PYTHON",
    os.path.join(SAM2_ENV_PREFIX, "bin", "python"),
)

# cfg 名称遵循 SAM2 官方 build_sam.py 约定
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# ckpt 统一放在 pretrained/，可通过环境变量覆盖
SAM2_CKPT = os.environ.get(
    "SAM2_CKPT",
    os.path.join(PROJECT_ROOT, "pretrained", "sam2.1_hiera_base_plus.pt"),
)

# 所有 I/O 统一放在：try/sam2/io/
SAM2_IO_ROOT = os.path.join(SAM2_DIR, "io")
os.makedirs(SAM2_IO_ROOT, exist_ok=True)


def sam2_refine_dynamic_mask(
    rgb_np: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    tmp_root: str = None,
    no_prompt: bool = False,
) -> np.ndarray:
    """
    用 sam2env 里的 run_sam2_once.py 对动态 mask 做 refinement.

    Args:
        rgb_np:   H x W x 3, uint8
        pos_mask: H x W, 0/1 或 bool, 动态 seed
        neg_mask: H x W, 0/1 或 bool, 静态 seed
        tmp_root: 若为 None，则使用 try/sam2/io/
        no_prompt: 若 True，命令行加 --no_prompt，run_sam2_once 忽略 seeds、只用全图 bbox。

    Returns:
        dyn_mask_sam2: H x W, bool
    """

    if tmp_root is None:
        tmp_root = SAM2_IO_ROOT
    os.makedirs(tmp_root, exist_ok=True)
    
    tmp_dir = os.path.join(tmp_root)
    os.makedirs(tmp_dir, exist_ok=True)

    rgb_path = os.path.join(tmp_dir, "rgb.png")
    pos_path = os.path.join(tmp_dir, "pos.npy")
    neg_path = os.path.join(tmp_dir, "neg.npy")
    out_path = os.path.join(tmp_dir, "dyn.npy")
    pos_png_path = os.path.join(tmp_dir, "pos.png")
    neg_png_path = os.path.join(tmp_dir, "neg.png")

    pos_png = (pos_mask.astype(np.uint8) * 255)
    neg_png = (neg_mask.astype(np.uint8) * 255)

    imageio.imwrite(pos_png_path, pos_png)
    imageio.imwrite(neg_png_path, neg_png)

    if rgb_np.dtype != np.uint8:
        rgb_save = np.clip(rgb_np, 0, 255).astype(np.uint8)
    else:
        rgb_save = rgb_np
    imageio.imwrite(rgb_path, rgb_save)

    np.save(pos_path, pos_mask.astype(np.uint8))
    np.save(neg_path, neg_mask.astype(np.uint8))

    # 优先用 conda 环境里的 python（便于传 PYTHONPATH），否则用 conda run
    if os.path.isfile(SAM2_PYTHON):
        cmd = [SAM2_PYTHON, SAM2_RUNNER_PY]
    else:
        cmd = ["conda", "run", "-p", SAM2_ENV_NAME, "python", SAM2_RUNNER_PY]
    cmd += [
        "--image", rgb_path,
        "--pos", pos_path,
        "--neg", neg_path,
        "--out", out_path,
        "--cfg", SAM2_CFG,
        "--ckpt", SAM2_CKPT,
        "--device", "cuda",
    ]
    if no_prompt:
        cmd.append("--no_prompt")

    env = os.environ.copy()
    env["PYTHONPATH"] = SAM2_DIR if not env.get("PYTHONPATH") else (SAM2_DIR + os.pathsep + env["PYTHONPATH"])

    try:
        result = subprocess.run(
            cmd,
            cwd=SAM2_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[SAM2] subprocess exited with {result.returncode}")
            if result.stderr:
                print(f"[SAM2] stderr:\n{result.stderr}")
            if result.stdout:
                print(f"[SAM2] stdout:\n{result.stdout}")
            return np.zeros_like(pos_mask, dtype=bool)
    except subprocess.TimeoutExpired:
        print("[SAM2] subprocess timed out (300s)")
        return np.zeros_like(pos_mask, dtype=bool)
    except subprocess.CalledProcessError as e:
        print(f"[SAM2] subprocess failed: {e}")
        return np.zeros_like(pos_mask, dtype=bool)

    if not os.path.exists(out_path):
        print(f"[SAM2] WARNING: {out_path} not found, return zeros")
        return np.zeros_like(pos_mask, dtype=bool)

    dyn = np.load(out_path).astype(bool)
    return dyn

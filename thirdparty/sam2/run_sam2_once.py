#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-shot SAM2 runner.

用法（由外部脚本调用）：
python sam2/run_sam2_once.py \
    --image /tmp/sam2_io/xxxx/rgb.png \
    --pos   /tmp/sam2_io/xxxx/pos.npy \
    --neg   /tmp/sam2_io/xxxx/neg.npy \
    --out   /tmp/sam2_io/xxxx/dyn.npy \
    --cfg   sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
    --ckpt  pretrained/sam2.1_hiera_base_plus.pt
"""

import argparse
import os

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------------------------------------------------
#  解析命令行
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Run SAM2 once given seeds")
    parser.add_argument("--image", type=str, required=True,
                        help="RGB image path (png/jpg)")
    parser.add_argument("--pos", type=str, required=True,
                        help="positive seed mask (.npy, HxW or flatten)")
    parser.add_argument("--neg", type=str, required=True,
                        help="negative seed mask (.npy, HxW or flatten)")
    parser.add_argument("--out", type=str, required=True,
                        help="output dynamic mask path (.npy, HxW)")
    parser.add_argument("--cfg", type=str, required=True,
                        help="SAM2 config name, e.g. sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="SAM2 checkpoint path, e.g. pretrained/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--no_prompt", action="store_true",
                        help="if set, ignore seeds and run SAM2 with full-image box only")
    parser.add_argument(
        "--no_box",
        action="store_true",
        help="if set, do not pass bbox from pos mask (points only; usually worse)",
    )
    parser.add_argument(
        "--debug_prompt",
        action="store_true",
        help="if set, save *_prompt_debug.png with points/box overlay",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=96,
        help="max positive/negative points sampled from each seed mask (default 96)",
    )
    return parser.parse_args()


# ---------------------------------------------------------
#  工具函数
# ---------------------------------------------------------
def load_image_rgb(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def fix_mask_shape(mask, H, W, name="mask"):
    """
    尝试把 mask 调整成 (H, W)，尽量不抛异常。
    支持：
      - HxW
      - WxH  (转置)
      - (H*W,)  一维展平
    其他情况：回退为全零。
    """
    # print("mask shape", mask.shape)
    mask = np.asarray(mask)

    # 已经是 HxW
    if mask.ndim == 2 and mask.shape == (H, W):
        return mask.astype(bool)

    # 可能是 WxH
    if mask.ndim == 2 and mask.shape == (W, H):
        print(f"[SAM2] {name}: transpose from {mask.shape} to ({H},{W})")
        return mask.T.astype(bool)

    # 可能是一维展平
    if mask.ndim == 1 and mask.size == H * W:
        print(f"[SAM2] {name}: reshape from 1D({mask.size}) to ({H},{W})")
        return mask.reshape(H, W).astype(bool)

    print(f"[SAM2] WARNING: {name} unexpected shape {mask.shape}, "
          f"fallback to zeros ({H},{W})")
    return np.zeros((H, W), dtype=bool)


def sample_points_from_mask(mask_bool, max_points=64):
    """
    从二值 mask 中均匀随机采样若干点作为 SAM2 提示。
    mask_bool: HxW bool
    返回：
        coords: Nx2 float32, (x, y)
    """
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    idx = np.arange(ys.size)
    if ys.size > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)

    coords = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return coords


def to_numpy(x):
    """兼容 torch.Tensor / numpy.ndarray"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ====================== 可视化函数 =========================
def visualize_all_instances(image_rgb, masks, alpha=0.45):
    """
    多实例可视化：
    - 每个 instance 随机亮色
    - 黑色轮廓

    image_rgb : H x W x 3, uint8 (RGB)
    masks     : N x H x W, bool/0-1
    """
    image_rgb = np.asarray(image_rgb)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    masks = masks.astype(bool)
    H, W, _ = image_rgb.shape
    N = masks.shape[0]

    vis = image_rgb.copy()

    # 彩色填充
    for i in range(N):
        m = masks[i]
        if not m.any():
            continue

        color = np.array([
            np.random.randint(80, 255),
            np.random.randint(80, 255),
            np.random.randint(80, 255),
        ], dtype=np.uint8)

        vis[m] = (
            vis[m].astype(np.float32) * (1.0 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)

    # 黑色轮廓
    for i in range(N):
        m_u8 = (masks[i].astype(np.uint8) * 255)
        if m_u8.sum() == 0:
            continue
        contours, _ = cv2.findContours(
            m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            cv2.drawContours(vis, [c], -1, (0, 0, 0), 2)

    return vis


def visualize_selected_mask(image_rgb, mask_bool, alpha=0.55):
    """
    把最终选择的 dyn mask 叠加在原图上（绿色+黑边），方便确认。
    """
    image_rgb = np.asarray(image_rgb)
    m = np.asarray(mask_bool).astype(bool)
    vis = image_rgb.copy()
    if not m.any():
        return vis

    color = np.array([40, 200, 80], dtype=np.uint8)

    vis[m] = (
        vis[m].astype(np.float32) * (1.0 - alpha)
        + color.astype(np.float32) * alpha
    ).astype(np.uint8)

    m_u8 = (m.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        cv2.drawContours(vis, [c], -1, (0, 0, 0), 2)

    return vis
# ===============================================================


# ---------------------------------------------------------
#  主逻辑（含：全部 instance 可视化 + 基于 seeds 的 union 策略）
# ---------------------------------------------------------
def main():
    args = parse_args()

    # 1. 加载图像
    image_rgb = load_image_rgb(args.image)  # HxWx3, uint8
    H, W, _ = image_rgb.shape

    # 2. 加载 / 构造 seeds + bbox
    if not args.no_prompt:
        # 正常模式：从 pos/neg.npy 里读种子
        pos_mask_raw = np.load(args.pos)
        neg_mask_raw = np.load(args.neg)

        pos_mask = fix_mask_shape(pos_mask_raw, H, W, "pos_mask")
        neg_mask = fix_mask_shape(neg_mask_raw, H, W, "neg_mask")

        # 使用 pos 的包围框作为 bbox（帮助 SAM2 扩张到整块 instance）
        ys, xs = np.where(pos_mask)
        if ys.size > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            pad = 12
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(W - 1, x1 + pad)
            y1 = min(H - 1, y1 + pad)
            box_np = np.array([[x0, y0, x1, y1]], dtype=np.float32)
        else:
            box_np = None
    else:
        pos_mask = np.zeros((H, W), dtype=bool)
        neg_mask = np.zeros((H, W), dtype=bool)
        box_np = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)
        print("[SAM2] no_prompt mode: using full-image box, ignoring pos/neg seeds")

    use_box = not args.no_box

    # 3. 采样提示点
    if not args.no_prompt:
        pos_pts = sample_points_from_mask(pos_mask, max_points=args.max_points)
        neg_pts = sample_points_from_mask(neg_mask, max_points=args.max_points)

        if pos_pts.shape[0] == 0 and box_np is None:
            dyn_mask = np.zeros((H, W), dtype=bool)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            np.save(args.out, dyn_mask)
            print(f"[SAM2] No positive seeds & no box, saved zero mask to: {args.out}")
            return

        pts = pos_pts
        labels = np.ones((pos_pts.shape[0],), dtype=np.int32)

        if neg_pts.shape[0] > 0:
            pts = np.concatenate([pts, neg_pts], axis=0)
            neg_labels = np.zeros((neg_pts.shape[0],), dtype=np.int32)
            labels = np.concatenate([labels, neg_labels], axis=0)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)

    # 4. 构建 SAM2 模型
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[SAM2] CUDA not available, fallback to CPU.")
        device = "cpu"

    # print(f"[SAM2] Loading model cfg={args.cfg}, ckpt={args.ckpt}")
    model = build_sam2(args.cfg, args.ckpt)
    model.to(device)
    model.eval()

    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image_rgb)

    pts_torch = torch.from_numpy(pts).to(device)
    labels_torch = torch.from_numpy(labels).to(device)

    if box_np is not None and use_box:
        if args.debug_prompt:
            debug_vis_path = args.out.replace(".npy", "_prompt_debug.png")
            vis = image_rgb.copy()
            ys, xs = np.where(pos_mask)
            for x, y in zip(xs[::20], ys[::20]):
                cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 255), -1)
            ys, xs = np.where(neg_mask)
            for x, y in zip(xs[::20], ys[::20]):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            x0, y0, x1, y1 = box_np[0].astype(int)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imwrite(debug_vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Saved SAM2 prompt visualization → {debug_vis_path}")
        box_torch = torch.from_numpy(box_np).to(device)
    else:
        box_torch = None

    # 5. SAM2 推理
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=pts_torch.unsqueeze(0) if pts_torch.numel() > 0 else None,
            point_labels=labels_torch.unsqueeze(0) if pts_torch.numel() > 0 else None,
            box=box_torch,
            multimask_output=True,
        )

    masks_np = to_numpy(masks)   # -> (1,M,H,W) or (M,H,W)
    scores_np = to_numpy(scores) # -> (1,M) or (M,)

    if masks_np.ndim == 4:
        # (1, M, H, W) → (M, H, W)
        masks_np = masks_np[0]
    elif masks_np.ndim == 3:
        # (M, H, W)
        pass
    elif masks_np.ndim == 2:
        # 只有一张 (H, W)，补一维 M=1
        masks_np = masks_np[None, ...]
    else:
        print(f"[SAM2] WARNING: unexpected masks ndim={masks_np.ndim}, fallback to zeros.")
        dyn_mask = np.zeros((H, W), dtype=bool)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, dyn_mask)
        return

    # ========= 可视化“全部 instance” =========
    try:
        vis_all = visualize_all_instances(image_rgb, masks_np)
        all_vis_path = args.out.replace(".npy", "_all_instances.png")
        cv2.imwrite(all_vis_path, cv2.cvtColor(vis_all, cv2.COLOR_RGB2BGR))
        # print(f"[SAM2] Saved all-instance visualization to: {all_vis_path}")
    except Exception as e:
        print(f"[SAM2] Failed to visualize all instances: {e}")
    # =================================================

    if scores_np.ndim == 2:
        scores_np = scores_np[0]
    scores_np = scores_np.astype(np.float32)

    M = masks_np.shape[0]
    pos_bool = pos_mask.astype(bool)
    neg_bool = neg_mask.astype(bool)

    # 6. 基于 seeds 的“多实例 union”策略
    eps = 1e-6
    pos_area = pos_bool.sum()
    dyn_mask = np.zeros((H, W), dtype=bool)

    sel_indices = []
    cov_thr = 0.22
    neg_thr = 0.55

    for i in range(M):
        m = to_numpy(masks_np[i])
        m_bin = m > 0.5
        area = m_bin.sum()
        if area == 0:
            continue

        pos_overlap = np.logical_and(m_bin, pos_bool).sum()
        neg_overlap = np.logical_and(m_bin, neg_bool).sum()

        if pos_area > 0:
            cov = pos_overlap / (pos_area + eps)
        else:
            cov = 0.0

        neg_ratio = neg_overlap / (area + eps)

        if (pos_area > 0 and cov >= cov_thr and neg_ratio <= neg_thr) or \
           (pos_area == 0 and neg_ratio <= neg_thr):
            dyn_mask |= m_bin
            sel_indices.append(i)

    # 如果一个都没选到，就退回到“单一 best”策略
    if len(sel_indices) == 0:
        best_idx = 0
        best_cov = -1.0
        best_bin = np.zeros((H, W), dtype=bool)

        for i in range(M):
            m = to_numpy(masks_np[i])
            m_bin = m > 0.5
            area = m_bin.sum()
            if area == 0:
                continue
            pos_overlap = np.logical_and(m_bin, pos_bool).sum()
            if pos_area > 0:
                cov = pos_overlap / (pos_area + eps)
            else:
                cov = 0.0
            if cov > best_cov:
                best_cov = cov
                best_idx = i
                best_bin = m_bin

        dyn_mask = best_bin.astype(bool)
        # print(f"[SAM2] union-select: none passed, fallback to best_idx={best_idx}, cov={best_cov:.3f}")
    else:
        # print(f"[SAM2] union-select: selected {len(sel_indices)} masks: {sel_indices}")
        pass

    # 7. 再次确保是 HxW
    dyn_mask = fix_mask_shape(dyn_mask, H, W, "dyn_mask")

    # ===================================================
    #   后处理：干掉边缘噪声 + 小块背景
    # ===================================================
    dyn_u8 = dyn_mask.astype(np.uint8)

    # (1) 如果有正样本，用其 bbox 扩张一圈，只保留 bbox 邻域里的前景
    # if (not args.no_prompt) and pos_mask.any():
    #     ys, xs = np.where(pos_mask)
    #     y0, y1 = ys.min(), ys.max()
    #     x0, x1 = xs.min(), xs.max()

    #     pad = 40  # 可调：范围太小就增大一点
    #     x0 = max(0, x0 - pad)
    #     y0 = max(0, y0 - pad)
    #     x1 = min(W - 1, x1 + pad)
    #     y1 = min(H - 1, y1 + pad)

    #     bbox_mask = np.zeros_like(dyn_u8, dtype=np.uint8)
    #     bbox_mask[y0:y1 + 1, x0:x1 + 1] = 1
    #     dyn_u8 = dyn_u8 * bbox_mask  # 只保留 bbox 邻域，远处的边缘背景直接砍掉

    # (2) 形态学闭运算：填缝、抹平内部小洞
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dyn_u8 = cv2.morphologyEx(dyn_u8, cv2.MORPH_CLOSE, k, iterations=1)

    # (3) 小连通块过滤：干掉零碎噪声块
    num, lab, stats, _ = cv2.connectedComponentsWithStats(dyn_u8)
    area_thr = H * W * 0.0025

    clean = np.zeros_like(dyn_u8)
    for L in range(1, num):  # 0 是背景
        if stats[L, cv2.CC_STAT_AREA] >= area_thr:
            clean[lab == L] = 1

    dyn_mask = clean.astype(bool)
    # ===================================================

    # 8. 保存结果
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, dyn_mask)

    # ========= 可视化“最终选中的 mask” =========
    try:
        vis_sel = visualize_selected_mask(image_rgb, dyn_mask)
        sel_vis_path = args.out.replace(".npy", "_selected.png")
        cv2.imwrite(sel_vis_path, cv2.cvtColor(vis_sel, cv2.COLOR_RGB2BGR))
        # print(f"[SAM2] Saved selected-mask visualization to: {sel_vis_path}")
    except Exception as e:
        print(f"[SAM2] Failed to visualize selected mask: {e}")


if __name__ == "__main__":
    main()

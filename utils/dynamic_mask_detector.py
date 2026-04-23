# -*- coding: utf-8 -*-
"""
动态掩码检测模块：结合光流一致性 + 时序融合 + 全局先验 + YOLO 先验，可选 SAM 精修。
每隔 N 个 keyframe 对当前窗口内的帧计算 flow-based 动态概率，做时序融合与全局投票后与 YOLO 合并，
得到最终 mask 并写回 Camera.motion_mask。
约定：motion_mask True=静态，False=动态。
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable


# 每多少 keyframe 触发一次检测；当前区间 6 帧 + 上一区间 6 帧 = 12 帧窗口
DETECT_INTERVAL = 6
WINDOW_SIZE = 12
CURRENT_BLOCK_SIZE = 6

# 光流不一致性视为动态的阈值 (mask_fwd 为一致性，1-mask_fwd 为不一致)
FLOW_DYNAMIC_THRESHOLD = 0.5
# 全局先验：静态/动态判定阈值
GLOBAL_STATIC_THR = 0.2
GLOBAL_DYNAMIC_THR = 0.6
# 时序融合 + 全局先验 与 单帧/融合 的权重（最终 p = (1 - global_weight) * p_fused + global_weight * p_global_at_t）
GLOBAL_PRIOR_WEIGHT = 0.3
# SAM 种子：光流不一致 + p_dyn 阈值（参考 4DGS DynamicSegConfig）
SAM_INCONS_THR = 0.4
SAM_DYN_THR = 0.3
SAM_NEG_INCONS_THR = 0.1
MIN_BLOB_AREA_RATIO = 0.002
FLOW_INCONS_LOCAL_K = 5
# DINO：flow seeds 百分位、语义“长成整块”面积阈值
DINO_FLOW_SEED_PERCENTILE = 95.0
DINO_REGION_SIM_PERCENTILE = 60
DINO_MIN_BLOB_AREA_RATIO = 0.002


@dataclass
class DynamicSegConfig:
    """与 4DGS DynamicSegConfig 保持一致。"""
    local_k: int = 5
    incons_percentile: float = 90.0
    min_blob_area_ratio: float = 0.002
    static_thresh: float = 0.2
    dynamic_thresh: float = 0.6
    sam2_incons_thr: float = 0.4
    sam2_dino_thr: float = 0.3
    sam2_neg_incons_thr: float = 0.1


# ============================================================================
# 工具函数：光流处理
# ============================================================================

def _normalized_flow_to_pixel(flow_normalized: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Camera 存的是 flow/ [W,H]*2，还原为像素位移 (dx,dy)，约定 source(x-dx,y-dy)->target(x,y)。"""
    # flow_normalized: HxWx2
    scale = torch.tensor([W, H], device=device, dtype=torch.float32) / 2.0
    return flow_normalized * scale.unsqueeze(0).unsqueeze(0)


def warp_with_flow(
    img: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    用光流将 2D 标量图从源帧 warp 到目标帧。
    约定: flow[y,x]=(dx,dy) 表示 目标帧(x,y) 来自 源帧(x-dx, y-dy)。
    img: H x W, flow: H x W x 2 (像素位移)。
    """
    assert img.dim() == 2 and flow.dim() == 3 and flow.size(2) == 2
    H, W = img.shape
    device = img.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    xx, yy = xx.float(), yy.float()
    x_src = xx - flow[..., 0]
    y_src = yy - flow[..., 1]
    x_norm = 2.0 * (x_src / max(W - 1, 1)) - 1.0
    y_norm = 2.0 * (y_src / max(H - 1, 1)) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
    img_b = img.unsqueeze(0).unsqueeze(0)
    warped = F.grid_sample(
        img_b, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return warped[0, 0]


def flow_local_inconsistency_torch(flow: torch.Tensor, k: int = FLOW_INCONS_LOCAL_K) -> torch.Tensor:
    """
    光流局部不一致性，参考 4DGS flow_local_inconsistency_torch。
    flow: H x W x 2 (像素位移 dx, dy)，返回 H x W [0,1]。
    """
    H, W, _ = flow.shape
    flow_2 = flow.permute(2, 0, 1).unsqueeze(0)
    kernel = torch.ones(2, 1, k, k, device=flow.device, dtype=flow.dtype) / (k * k)
    local_avg = F.conv2d(flow_2, kernel, padding=k // 2, groups=2)
    diff = flow_2 - local_avg
    mag = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + 1e-12)
    mag = mag[0] * 50
    m_min, m_max = mag.min(), mag.max()
    incons = (mag - m_min) / (m_max - m_min + 1e-6)
    return incons


# ============================================================================
# DINO 特征提取
# ============================================================================

def build_flow_seeds(
    avg_flow: torch.Tensor,
    percentile: float = DINO_FLOW_SEED_PERCENTILE,
    min_blob_ratio: float = MIN_BLOB_AREA_RATIO,
    border_ignore: int = 10,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    参考 4DGS build_flow_seeds：根据光流不一致性构造 flow seeds。
    返回 (pos_seed_flow, seed_raw_np, seed_big_np)，与 4DGS 一致。
    """
    incons = avg_flow
    device = incons.device
    H, W = incons.shape
    inc = incons.detach().cpu().numpy().astype(np.float32)
    if border_ignore > 0:
        inc[:border_ignore, :] = 0
        inc[-border_ignore:, :] = 0
        inc[:, :border_ignore] = 0
        inc[:, -border_ignore:] = 0
    thr = np.percentile(inc, percentile)
    mask = (inc > thr).astype(np.uint8)
    seed_raw_np = mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    area_thr = H * W * float(min_blob_ratio)
    big = np.zeros_like(mask, dtype=np.uint8)
    for L in range(1, num):
        if stats[L, cv2.CC_STAT_AREA] >= area_thr:
            big[lab == L] = 1
    seed_big_np = big.copy()
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(big, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    pos_seed_flow = torch.from_numpy(filled.astype(bool)).to(device)
    return pos_seed_flow, seed_raw_np, seed_big_np


def build_dino_similarity(
    dino_feat: torch.Tensor,
    pos_seed: torch.Tensor,
    out_shape: Tuple[int, int],
    device: torch.device,
    apply_bilateral: bool = True,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 0.1,
    bilateral_sigma_space: float = 5.0,
) -> torch.Tensor:
    """
    参考 4DGS build_dino_similarity：用 flow seeds 在 DINO 特征上做 prototype，得到相似度图并上采样到 out_shape。
    """
    C, Hf, Wf = dino_feat.shape
    if isinstance(pos_seed, np.ndarray):
        pos_seed_t = torch.from_numpy(pos_seed.astype(bool)).to(device)
    else:
        pos_seed_t = pos_seed.to(device).float()
    seed_s = F.interpolate(
        pos_seed_t.unsqueeze(0).unsqueeze(0), (Hf, Wf), mode="nearest"
    )[0, 0].bool()
    if seed_s.sum() < 5:
        fn = torch.norm(dino_feat, dim=0)
        fn = (fn - fn.min()) / (fn.max() - fn.min() + 1e-6)
        sim_s = fn
    else:
        feat_f = dino_feat.view(C, -1)
        mask_f = seed_s.view(-1)
        proto = feat_f[:, mask_f].mean(1, keepdim=True)
        fa = F.normalize(feat_f.t(), dim=1)
        pn = F.normalize(proto[:, 0], dim=0)
        sim = (fa @ pn).view(Hf, Wf)
        sim_s = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)
    sim_full = F.interpolate(
        sim_s.unsqueeze(0).unsqueeze(0),
        out_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    if apply_bilateral:
        try:
            sim_np = sim_full.detach().float().cpu().numpy().astype(np.float32)
            sim_np = cv2.bilateralFilter(
                sim_np,
                d=int(bilateral_d),
                sigmaColor=float(bilateral_sigma_color),
                sigmaSpace=float(bilateral_sigma_space),
            )
            sim_full = torch.from_numpy(sim_np).to(device=device, dtype=sim_full.dtype)
        except Exception:
            pass

    return sim_full


def build_p_dyn_flow_dino(
    sim_full: torch.Tensor,
    pos_seed_np: np.ndarray,
    device: torch.device,
    sim_percentile: float = DINO_REGION_SIM_PERCENTILE,
    min_blob_ratio: float = DINO_MIN_BLOB_AREA_RATIO,
) -> torch.Tensor:
    """
    参考 4DGS：在 flow seeds 邻域内用 DINO 相似度做自适应阈值 + 连通块，得到 p_dyn_flow_dino。
    """
    H, W = sim_full.shape
    pos_seed_np = pos_seed_np.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    region_np = cv2.dilate(pos_seed_np, k, iterations=1)
    sim_np = sim_full.detach().cpu().numpy().astype(np.float32)
    mask_region = region_np > 0
    if mask_region.sum() > 50:
        thr = np.percentile(sim_np[mask_region], sim_percentile)
    else:
        thr = np.percentile(sim_np, 70)
    mask_high = (sim_np >= thr).astype(np.uint8) * region_np
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_high = cv2.morphologyEx(mask_high, cv2.MORPH_CLOSE, k2, iterations=1)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask_high)
    clean = np.zeros_like(mask_high, dtype=np.uint8)
    area_thr = H * W * min_blob_ratio
    for L in range(1, num):
        if stats[L, cv2.CC_STAT_AREA] >= area_thr:
            clean[lab == L] = 1
    p_dyn = torch.from_numpy(clean.astype(np.float32)).to(device) * sim_full
    return p_dyn.clamp(0.0, 1.0)


# ============================================================================
# SAM 相关函数
# ============================================================================

def _get_sam_refine_fn() -> Optional[Callable]:
    """与 4DGS 一致：使用 utils.sam2_bridge.sam2_refine_dynamic_mask。"""
    try:
        from utils.sam2_bridge import sam2_refine_dynamic_mask
        return sam2_refine_dynamic_mask
    except Exception:
        return None


def _get_dino_extractor(device: torch.device, backbone_name: str = "dinov2_vitb14"):
    """与 4DGS 一致：使用 utils.extract_dino.DinoFeatureExtractor（相同归一化与 get_intermediate_layers）。"""
    if getattr(_get_dino_extractor, "_extractor", None) is not None:
        return _get_dino_extractor._extractor
    try:
        from utils.extract_dino import DinoFeatureExtractor
        ext = DinoFeatureExtractor(backbone_name=backbone_name, pretrained=False, freeze_backbone=True)
        ext.to(device).eval()
        _get_dino_extractor._extractor = ext
        return ext
    except Exception:
        _get_dino_extractor._extractor = None
        return None


def get_dino_feat_for_image(image: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
    """
    与 4DGS 一致：用 DinoFeatureExtractor 取特征，输入 [0,1]，归一化 123.675/255 等，输出 [C, Hf, Wf]。
    """
    ext = _get_dino_extractor(device)
    if ext is None:
        return None
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        feat = ext.forward(image)
    return feat[0] if feat.dim() == 4 else feat


def select_prompt_points_kmeans(mask_np: np.ndarray, k: int = 4, min_dist: float = 5.0) -> List[Tuple[int, int]]:
    """
    修改版：确保选出的点至少距离边缘 min_dist 像素。
    """
    # 1. 计算距离变换
    # dist_map 中的每个像素值代表其到最近背景（0）的距离
    dist_map = cv2.distanceTransform(mask_np.astype(np.uint8), cv2.DIST_L2, 5)
    
    # 2. 筛选：只保留距离边缘足够远的像素
    # 获取掩码，只选取距离边缘 > min_dist 的点
    valid_mask = (dist_map > min_dist)
    
    # 如果筛选后没有点，则放宽条件，回退到原始所有前景点
    if np.sum(valid_mask) == 0:
        ys, xs = np.where(mask_np > 0)
    else:
        ys, xs = np.where(valid_mask)
        
    num_fg = len(xs)
    if num_fg == 0:
        return []
        
    # 3. K-means 逻辑保持不变
    K = min(k, num_fg)
    samples = np.stack([xs, ys], axis=1).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(samples, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    
    points = []
    for ci in range(K):
        mask_ci = (labels.ravel() == ci)
        cluster_pts = samples[mask_ci]
        if cluster_pts.shape[0] == 0: continue
        
        # 找到聚类内距离该中心最近的点
        dists = np.sum((cluster_pts - centers[ci]) ** 2, axis=1)
        best_xy = cluster_pts[np.argmin(dists)]
        points.append((int(best_xy[0]), int(best_xy[1])))
        
    return points


def build_sam2_seeds_from_flow_dino(
    incons: torch.Tensor,
    p_dyn: torch.Tensor,
    cfg: DynamicSegConfig,
    rgb: Optional[torch.Tensor] = None,
    debug_dir: Optional[str] = None,
    frame_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    与 4DGS build_sam2_seeds_from_flow_dino 一致：根据 flow inconsistency + p_dyn(flow+DINO) 构造 SAM2 正负 seeds。
    返回 (pos_seed_t, neg_seed_t) torch.BoolTensor。
    """
    device = incons.device
    H, W = incons.shape
    incons_np = incons.detach().cpu().numpy()
    p_dyn_np = p_dyn.detach().cpu().numpy()
    border = 10
    if border > 0:
        incons_np[:border, :] = 0
        incons_np[-border:, :] = 0
        incons_np[:, :border] = 0
        incons_np[:, -border:] = 0
        p_dyn_np[:border, :] = 0
        p_dyn_np[-border:, :] = 0
        p_dyn_np[:, :border] = 0
        p_dyn_np[:, -border:] = 0
    raw_pos = np.logical_and(
        incons_np > cfg.sam2_incons_thr,
        p_dyn_np > cfg.sam2_dino_thr,
    ).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed_clean = cv2.morphologyEx(raw_pos, cv2.MORPH_OPEN, k3, iterations=1)
    seed_clean = cv2.morphologyEx(seed_clean, cv2.MORPH_CLOSE, k3, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed_clean, connectivity=8)
    area_thr = H * W * cfg.min_blob_area_ratio
    big = np.zeros_like(seed_clean)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= area_thr:
            big[labels == lbl] = 255
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(big)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    pos_seed_np = filled > 0
    if pos_seed_np.sum() < 50:
        pos_seed_np = (p_dyn_np > 0.6)
    neg_seed_np = np.logical_and(incons_np < cfg.sam2_neg_incons_thr, np.logical_not(pos_seed_np))
    pos_seed_t = torch.from_numpy(pos_seed_np).to(device=device, dtype=torch.bool)
    neg_seed_t = torch.from_numpy(neg_seed_np).to(device=device, dtype=torch.bool)
    return pos_seed_t, neg_seed_t


def run_sam2_with_flow_dino(
    vp_t: Any,
    incons: torch.Tensor,
    p_dyn_flow_dino: torch.Tensor,
    cfg: DynamicSegConfig,
    device: torch.device,
    save_dir: Optional[str] = None,
    frame_idx: int = 0,
    sam_refine_fn: Optional[Callable] = None,
    yolo_dynamic: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    flow+DINO seeds + YOLO 一起作为 SAM 的输入：先 build_sam2_seeds(光流+DINO)，再与 YOLO 动态区域合并 → K 点稀疏 → SAM2。
    返回 (sam2_mask_t, p_dyn_sam2)。
    """
    if sam_refine_fn is None:
        sam_refine_fn = _get_sam_refine_fn()
    if sam_refine_fn is None:
        return (p_dyn_flow_dino > 0.5).to(torch.bool), p_dyn_flow_dino
    pos_seed_t, neg_seed_t = build_sam2_seeds_from_flow_dino(
        incons=incons, p_dyn=p_dyn_flow_dino, cfg=cfg,
        rgb=vp_t.original_image, debug_dir=save_dir, frame_idx=frame_idx,
    )
    pos_seed_np = pos_seed_t.cpu().numpy().astype(np.uint8)
    # 光流 + YOLO 一起作为 SAM 正样本：合并 YOLO 检测到的动态区域
    if yolo_dynamic is not None:
        yolo_np = yolo_dynamic.detach().cpu().numpy().astype(np.uint8)
        if yolo_np.shape != pos_seed_np.shape:
            yolo_np = cv2.resize(yolo_np, (pos_seed_np.shape[1], pos_seed_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        pos_seed_np = np.maximum(pos_seed_np, yolo_np)
    # 使用 neg_seed_t 作为 SAM 负样本（低光流不一致且非正样本区域）
    neg_seed_np = neg_seed_t.cpu().numpy().astype(np.uint8)
    if neg_seed_np.shape != pos_seed_np.shape:
        neg_seed_np = cv2.resize(neg_seed_np, (pos_seed_np.shape[1], pos_seed_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 负样本与正样本互斥
    neg_seed_np = np.minimum(neg_seed_np, 1 - np.minimum(pos_seed_np, 1))
    border = 20
    if border > 0:
        pos_seed_np[:border, :] = 0
        pos_seed_np[-border:, :] = 0
        pos_seed_np[:, :border] = 0
        pos_seed_np[:, -border:] = 0
        neg_seed_np[:border, :] = 0
        neg_seed_np[-border:, :] = 0
        neg_seed_np[:, :border] = 0
        neg_seed_np[:, -border:] = 0
    prompt_points = select_prompt_points_kmeans(pos_seed_np, k=4)
    H, W = pos_seed_np.shape
    pos_seed_np = np.zeros((H, W), dtype=np.uint8)
    for (x, y) in prompt_points:
        pos_seed_np[y, x] = 1
    img_t = vp_t.original_image
    img_np = img_t.permute(1, 2, 0).cpu().numpy()
    img_u8 = (img_np * 255.0).clip(0, 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    # 使用项目根目录下的相对路径：<project_root>/thirdparty/sam2/io
    project_root = os.path.dirname(os.path.dirname(__file__))
    tmp_root = os.path.join(project_root, "thirdparty", "sam2", "io")
    
    # 暂不使用 neg_seed_np，负样本传全零
    neg_for_sam = np.zeros_like(pos_seed_np, dtype=np.uint8)
    if False:
        sam2_mask_np = sam_refine_fn(rgb_np=img_u8, pos_mask=pos_seed_np, neg_mask=neg_for_sam, tmp_root=tmp_root, no_prompt=False)
        sam2_mask_t = torch.from_numpy(sam2_mask_np.astype(bool)).to(device)
        sam2_prob = sam2_mask_t.float()
        p_dyn_sam2 = ((yolo_dynamic + sam2_prob) / 2.0).clamp(0.0, 1.0)
        return sam2_prob, p_dyn_sam2
    else:
        p_dyn_sam2 = ((yolo_dynamic * 2) / 2.0).clamp(0.0, 1.0)
        return None, p_dyn_sam2


# ============================================================================
# 核心流程：单帧动态概率计算
# ============================================================================

def compute_p_dyn_single(
    vp: Any,
    flow_t: torch.Tensor,
    avg_flow: torch.Tensor,
    cfg: DynamicSegConfig,
    device: torch.device,
    save_dir: Optional[str] = None,
    frame_idx: int = 0,
    video_idx: Optional[int] = None,
    frame_reader: Optional[Any] = None,
    sam_refine_fn: Optional[Callable] = None,
    seg_chair: bool = False,
) -> torch.Tensor:
    """
    与 4DGS compute_p_dyn_single 一致：单帧 flow inconsistency → flow seeds → DINO similarity
    → p_dyn_flow_dino（长成整块）→ SAM2 → p_dyn_final。返回该帧动态概率 p_dyn。
    """
    H, W = vp.original_image.shape[1], vp.original_image.shape[2]
    # # 1) flow inconsistency
    # incons = flow_local_inconsistency_torch(flow_t, cfg.local_k)
    # 2) flow seeds（用于 DINO）
    pos_seed_flow, seed_raw_np, seed_big_np = build_flow_seeds(
        avg_flow,
        percentile=cfg.incons_percentile,
        min_blob_ratio=cfg.min_blob_area_ratio,
    )
    # 3) DINO similarity（与 4DGS 一致：vp.dino_feat 或 extract_dino 提取，相同归一化）
    dino_feat = getattr(vp, "dino_feat", None)
    if dino_feat is None:
        dino_feat = get_dino_feat_for_image(vp.original_image, device)
    if dino_feat is not None:
        seed_for_dino = torch.from_numpy(seed_big_np.astype(bool)).to(device) if isinstance(seed_big_np, np.ndarray) else pos_seed_flow
        sim_full = build_dino_similarity(dino_feat, seed_for_dino, (H, W), device)
    else:
        sim_full = torch.ones(H, W, device=device, dtype=torch.float32) * 0.5
    # 4) 用 DINO 语义“长成整块”
    pos_seed_np = pos_seed_flow.detach().cpu().numpy().astype(np.uint8)
    p_dyn_flow_dino = build_p_dyn_flow_dino(sim_full, pos_seed_np, device)
    # 5) SAM2：光流+DINO 与 YOLO 一起作为 SAM 输入（seeds = flow+DINO ∪ YOLO → K 点 → sam2_refine_dynamic_mask）
    yolo_dynamic = None
    if frame_reader is not None and video_idx is not None:
        yolo_dynamic = get_yolo_dynamic_mask(frame_reader, video_idx, vp.original_image, device, seg_chair=seg_chair)
    try:
        _, p_dyn_sam2 = run_sam2_with_flow_dino(
            vp_t=vp,
            incons=avg_flow,
            p_dyn_flow_dino=p_dyn_flow_dino,
            cfg=cfg,
            device=device,
            save_dir=save_dir,
            frame_idx=frame_idx,
            sam_refine_fn=sam_refine_fn,
            yolo_dynamic=yolo_dynamic,
        )
        return p_dyn_sam2
    except Exception as e:
        print(f"{e}")
        pass
    return p_dyn_flow_dino

def _flow_avg_motion(f: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """归一化光流 [H,W,2] 或 [1,H,W,2] -> 有限差分梯度幅值全图最大值。异常或非法形状时返回 0。"""
    try:
        if not isinstance(f, torch.Tensor):
            f = torch.from_numpy(np.asarray(f)).float().to(dev)
        f = f.detach().to(dev)
        while f.dim() > 3:
            f = f.squeeze(0)
        if f.dim() != 3 or f.shape[2] != 2 or f.shape[0] < 2 or f.shape[1] < 2:
            return torch.tensor(0.0, device=dev, dtype=torch.float32)
        f_in = f.permute(2, 0, 1).unsqueeze(0)
        df_dx = f_in[:, :, :, 1:] - f_in[:, :, :, :-1]  # -> (1,2,H,W-1)
        df_dy = f_in[:, :, 1:, :] - f_in[:, :, :-1, :]  # -> (1,2,H-1,W)
        df_dx = torch.nn.functional.pad(df_dx, (0, 1, 0, 0))
        df_dy = torch.nn.functional.pad(df_dy, (0, 0, 0, 1))
        grad_mag = torch.sqrt((df_dx ** 2).sum(dim=1) + (df_dy ** 2).sum(dim=1) + 1e-8)
        # print(f"Flow avg motion (max grad mag): {grad_mag.max().item():.6f}")
        return grad_mag
    except Exception:
        print("error in _flow_avg_motion")
        return torch.tensor(0.0, device=dev, dtype=torch.float32)

# ============================================================================
# 核心流程：动态分割流水线
# ============================================================================

def dynamic_segmentation_pipeline(
    viewpoints: List[Any],
    device: torch.device,
    local_k: int = 5,
    incons_percentile: float = 90.0,
    min_blob_area_ratio: float = 0.001,
    static_thresh: float = 0.2,
    dynamic_thresh: float = 0.6,
    save_dir: Optional[str] = None,
    video_idxs: Optional[List[int]] = None,
    frame_reader: Optional[Any] = None,
    sam_refine_fn: Optional[Callable] = None,
    seg_chair: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    与 4DGS dynamic_segmentation_pipeline 一致：
    1) 相邻帧光流 2) 单帧 p_dyn (compute_p_dyn_single) 3) 时序融合 4) 全局先验
    返回 p_single, p_fused, p_ref, static_ref, dyn_ref。
    """
    T = len(viewpoints)
    assert T >= 1
    if video_idxs is None:
        video_idxs = list(range(T))
    cfg = DynamicSegConfig(
        local_k=local_k,
        incons_percentile=incons_percentile,
        min_blob_area_ratio=min_blob_area_ratio,
        static_thresh=static_thresh,
        dynamic_thresh=dynamic_thresh,
    )
    # ----- 0. 为每个 camera 设置 DINO 特征属性 dino_feat：已有则跳过，否则计算并写入，避免重复计算
    dino_ext = _get_dino_extractor(device)
    if dino_ext is not None:
        for vp in viewpoints:
            dino_feat = getattr(vp, "dino_feat", None)
            if dino_feat is not None:
                continue
            if getattr(vp, "original_image", None) is None:
                continue
            with torch.no_grad():
                img = vp.original_image.unsqueeze(0).to(device)
                vp.dino_feat = dino_ext.forward(img)[0]
    # ----- 1. 光流（t 与 t+5，超出则 (t+5)%T 从开头选） -----
    flows_fwd: List[torch.Tensor] = []
    flows_bwd: List[torch.Tensor] = []
    cons_fwd: List[torch.Tensor] = []
    cons_bwd: List[torch.Tensor] = []
    avgs_fwd: List[torch.Tensor] = []
    # 光流用 t 与 t+5 配对（t+5 超出则从开头选 (t+5)%T），保持 T-1 对以兼容后续时序融合
    for t in range(T):
        t_next = t + 5 if t + 5 < T else t - 5
        if t_next == t:
            t_next = (t + 1) % T  # 避免自配对
        vp_t, vp_tp5 = viewpoints[t], viewpoints[t_next]
        try:
            vi_next = video_idxs[t_next] if t_next < len(video_idxs) else t_next
            flow_n, flow_back_n, mask_fwd, mask_bwd, _ = vp_tp5.generate_flow(
                vp_tp5.original_image, vi_next, vp_t.original_image
            )
        except Exception:
            break
        h, w = vp_tp5.original_image.shape[1], vp_tp5.original_image.shape[2]
        flow_pix = _normalized_flow_to_pixel(flow_n, h, w, device)
        flow_back_pix = _normalized_flow_to_pixel(flow_back_n, h, w, device)
        m_fwd = mask_fwd.float() if mask_fwd is not None else torch.ones(h, w, device=device)
        m_bwd = mask_bwd.float() if mask_bwd is not None else torch.ones(h, w, device=device)
        
        # --- 光流运动检测：光流梯度幅值判断是否有运动 ---
        motion_threshold = 0.025
        avg_fwd = _flow_avg_motion(flow_n, device)
        avgs_fwd.append(avg_fwd.squeeze(0))
        has_flow_motion = (avg_fwd.max().detach() >= motion_threshold).item()
        if not has_flow_motion:
            flow_pix = torch.zeros(h, w, 2, device=device, dtype=flow_pix.dtype)
            flow_back_pix = torch.zeros(h, w, 2, device=device, dtype=flow_back_pix.dtype)
            m_fwd = torch.zeros(h, w, device=device, dtype=m_fwd.dtype)
            m_bwd = torch.zeros(h, w, device=device, dtype=m_bwd.dtype)

        flows_fwd.append(flow_pix)
        flows_bwd.append(flow_back_pix)
        cons_fwd.append(m_fwd.to(device))
        cons_bwd.append(m_bwd.to(device))
        
    if len(flows_fwd) < 1 and T >= 2:
        h, w = viewpoints[0].original_image.shape[1], viewpoints[0].original_image.shape[2]
        empty = torch.zeros(h, w, device=device, dtype=torch.float32)
        return [], [], empty, empty.bool(), empty.bool()
    # ----- 2. 单帧 p_dyn -----
    p_single: List[torch.Tensor] = []
    for t in range(T):
        flow_t = flows_fwd[t]
        vi = video_idxs[t] if t < len(video_idxs) else t
        
        # 检查 flow_t 是否为全零 tensor
        if flow_t.abs().max() < 1e-6:
            # 如果 flow 为全零，直接返回全零的 p_dyn
            H, W = viewpoints[t].original_image.shape[1], viewpoints[t].original_image.shape[2]
            p_dyn_t = torch.zeros(H, W, device=device, dtype=torch.float32)
        else:
            p_dyn_t = compute_p_dyn_single(
                viewpoints[t], flow_t, avgs_fwd[t], cfg, device,
                save_dir=save_dir, frame_idx=t, video_idx=vi,
                frame_reader=frame_reader, sam_refine_fn=sam_refine_fn, seg_chair=seg_chair,
            )
        p_single.append(p_dyn_t)
    # ----- 3. 时序融合 -----
    if len(p_single) < 2:
        p_fused = p_single
    else:
        p_fused = temporal_fuse_p_dyn_refined(p_single, flows_fwd, flows_bwd, cons_fwd, cons_bwd)
    # # ----- 4. 全局先验 -----
    # ref_t = len(p_fused) // 2
    # p_ref, static_ref, dyn_ref = compute_global_prior(
    #     p_fused, flows_fwd, flows_bwd, cons_fwd, cons_bwd,
    #     ref_t=ref_t, static_thr=static_thresh, dyn_thr=dynamic_thresh,
    # )
    # return p_single, p_fused, p_ref, static_ref, dyn_ref
    
    return p_single, p_fused, _, _, _

# ============================================================================
# 辅助函数：前端接口支持
# ============================================================================

def _make_minimal_viewpoint(frame_reader: Any, stream_idx: int, device: torch.device, ref_camera: Any) -> Any:
    """从 frame_reader 取一帧，构造仅含 original_image 与 generate_flow 的轻量 viewpoint，用于「所有帧」窗口。"""
    try:
        item = frame_reader[stream_idx]
        if isinstance(item, (list, tuple)):
            # 常见：return (index, color_data, depth_data, pose, ...)，image 为 item[1]
            image = item[1]
        else:
            image = getattr(item, "image", getattr(item, "original_image", item))
        if image is None:
            return None
        if not torch.is_tensor(image):
            image = torch.from_numpy(np.asarray(image)).float()
        image = image.to(device)
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() != 3 or image.shape[0] not in (1, 3):
            return None
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        # 绑定 ref_camera.generate_flow，供相邻帧光流用
        class _MinimalVp:
            pass
        vp = _MinimalVp()
        vp.original_image = image
        vp.uid = stream_idx
        vp.dino_feat = None
        vp.motion_mask = None
        vp.generate_flow = lambda img_next, vi, img_prev: ref_camera.generate_flow(img_next, vi, img_prev)
        return vp
    except Exception:
        return None


def _warp_flow_np(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """numpy 光流 warp，与 Camera.warp_flow 一致。"""
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]
    return cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def _compute_fwdbwd_mask_np(fwd_flow: np.ndarray, bwd_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """光流前后向一致性掩码，与 Camera.compute_fwdbwd_mask 一致。"""
    alpha_1, alpha_2 = 0.5, 0.5
    bwd2fwd = _warp_flow_np(bwd_flow, fwd_flow)
    fwd_lr = np.linalg.norm(fwd_flow + bwd2fwd, axis=-1)
    fwd_mask = fwd_lr < alpha_1 * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd, axis=-1)) + alpha_2
    fwd2bwd = _warp_flow_np(fwd_flow, bwd_flow)
    bwd_lr = np.linalg.norm(bwd_flow + fwd2bwd, axis=-1)
    bwd_mask = bwd_lr < alpha_1 * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd, axis=-1)) + alpha_2
    return fwd_mask, bwd_mask


_raft_models: Dict[str, Any] = {}


def _compute_flow_two_images(
    img_prev: torch.Tensor,
    img_next: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """
    仅用两帧图像计算光流，不依赖 Camera。返回与 generate_flow 相同格式：
    (flow_n, flow_back_n, mask_fwd, mask_bwd, None)，flow 为归一化形式。
    """
    global _raft_models
    dev_key = str(device)
    if dev_key not in _raft_models:
        try:
            from RAFT.raft import RAFT
            from RAFT.utils.utils import InputPadder
            class _Args:
                model = "pretrained/raft-things.pth"
                small = False
                mixed_precision = False
            args = _Args()
            model = torch.nn.DataParallel(RAFT(args))
            model.load_state_dict(torch.load(args.model))
            model = model.module.to(device).eval()
            _raft_models[dev_key] = (model, InputPadder)
        except Exception:
            return None, None, None, None, None
    model, InputPadder = _raft_models[dev_key]
    with torch.no_grad():
        a = img_prev.detach().clone().to(device) * 255.0
        b = img_next.detach().clone().to(device) * 255.0
        if a.dim() == 3:
            a, b = a[None], b[None]
        padder = InputPadder(a.shape)
        a, b = padder.pad(a, b)
        _, flow_fwd = model(a, b, iters=20, test_mode=True)
        _, flow_bwd = model(b, a, iters=20, test_mode=True)
        flow_fwd = padder.unpad(flow_fwd[0]).permute(1, 2, 0).cpu().numpy()
        flow_bwd = padder.unpad(flow_bwd[0]).permute(1, 2, 0).cpu().numpy()
        mask_fwd, mask_bwd = _compute_fwdbwd_mask_np(flow_fwd, flow_bwd)
        H, W = flow_fwd.shape[0], flow_fwd.shape[1]
        scale = torch.tensor([W, H], device=device, dtype=torch.float32) / 2.0
        flow_n = torch.from_numpy(flow_fwd).float().to(device) / scale.unsqueeze(0).unsqueeze(0) * 2.0
        flow_back_n = torch.from_numpy(flow_bwd).float().to(device) / scale.unsqueeze(0).unsqueeze(0) * 2.0
        m_fwd = torch.from_numpy(mask_fwd).float().to(device)
        m_bwd = torch.from_numpy(mask_bwd).float().to(device)
        if m_fwd.dim() == 3:
            m_fwd = m_fwd.squeeze(0)
        if m_bwd.dim() == 3:
            m_bwd = m_bwd.squeeze(0)
    return flow_n, flow_back_n, m_fwd, m_bwd, None


def _stream_item_to_image(item: Any, device: torch.device) -> Optional[torch.Tensor]:
    """从 stream[i] 取出图像并转为 [C,H,W] tensor，[0,1]。"""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    image = item[1]
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image)).float()
    image = image.to(device)
    if image.dim() == 4:
        image = image.squeeze(0)
    if image.dim() != 3 or image.shape[0] not in (1, 3):
        return None
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


# ============================================================================
# 前端接口函数
# ============================================================================

def run_frontend_motion_mask_and_save(
    stream: Any,
    keyframe_timestamps: List[int],
    keyframe_video_idxs: List[int],
    save_dir: str,
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    前端专用：仅用 stream 获得的数据跑动态分割并保存掩码图片，不依赖 Camera/viewpoint。
    窗口：当前关键帧到前两个关键帧之间区间内的所有帧（两个区间）；若只有两个关键帧则只选一个区间。
    将 motion_mask 按流式索引保存为 {save_dir}/motion_mask/{timestamp:06d}.png。
    """
    config = config or {}
    if not config.get("enable_dynamic_mask_detection", True):
        return False
    if len(keyframe_timestamps) < 2 or len(keyframe_video_idxs) < 2:
        return False

    first_ts = keyframe_timestamps[0]
    last_ts = keyframe_timestamps[-1]
    all_indices = list(range(first_ts, last_ts + 1))

    # 从 stream 按 all_indices 取每一帧，构造轻量“帧”对象（记录实际参与计算的流式索引）
    frames: List[Any] = []
    frame_stream_indices: List[int] = []
    for ts in all_indices:
        item = stream[ts]
        image = _stream_item_to_image(item, device)
        if image is None:
            continue
        obj = type("StreamFrame", (), {})()
        obj.original_image = image
        obj.uid = ts
        obj.dino_feat = None
        obj.motion_mask = None
        obj.generate_flow = lambda img_next, vi, img_prev, d=device: _compute_flow_two_images(img_prev, img_next, d)
        frames.append(obj)
        frame_stream_indices.append(ts)

    # video_idxs 与帧一一对应，用流式索引（frame_reader 按 index 读）
    video_idxs_for_pipeline = frame_stream_indices
    static_thr = config.get("global_static_thr", GLOBAL_STATIC_THR)
    cfg_merged = {
        "save_dir": save_dir,
        "enable_motion_mask_update": True,
        "global_static_thr": static_thr,
        "global_dynamic_thr": config.get("global_dynamic_thr", GLOBAL_DYNAMIC_THR),
        "seg_chair": config.get("seg_chair", False),
    }

    with torch.no_grad():
        p_single, p_fused, p_ref, static_ref, dyn_ref = dynamic_segmentation_pipeline(
            frames,
            device,
            local_k=config.get("local_k", FLOW_INCONS_LOCAL_K),
            incons_percentile=config.get("incons_percentile", DINO_FLOW_SEED_PERCENTILE),
            min_blob_area_ratio=config.get("min_blob_area_ratio", MIN_BLOB_AREA_RATIO),
            static_thresh=static_thr,
            dynamic_thresh=cfg_merged["global_dynamic_thr"],
            save_dir=save_dir,
            video_idxs=video_idxs_for_pipeline,
            frame_reader=stream,
            sam_refine_fn=config.get("sam_refine_fn") or _get_sam_refine_fn(),
            seg_chair=cfg_merged["seg_chair"],
        )

    if len(p_single) < 1:
        return False

    os.makedirs(os.path.join(save_dir, "motion_mask"), exist_ok=True)
    for i in range(len(p_single)):
        if i >= len(frame_stream_indices):
            break
        save_idx = frame_stream_indices[i]
        motion_mask = p_single[i]<=0
        _save_motion_mask_by_idx(motion_mask, save_idx, cfg_merged)
    return True


def build_window_indices(
    keyframe_video_idxs: List[int],
    window_size: int = WINDOW_SIZE,
    current_size: int = CURRENT_BLOCK_SIZE,
    keyframe_idxs: Optional[List[int]] = None,
) -> tuple:
    """
    若 keyframe_idxs 未传：从 keyframe 列表取「上一区间+当前区间」的 video_idx（仅关键帧）。
    若 keyframe_idxs 已传：all_indices 为首尾之间的所有流式帧索引 [first_stream, ..., last_stream]，current_indices 仍为当前块内的 keyframe video_idx。
    """
    n = len(keyframe_video_idxs)
    if n == 0:
        return [], []
    if keyframe_idxs is not None and len(keyframe_idxs) == n:
        # 使用首尾之间的所有帧（流式索引）
        first_stream = keyframe_idxs[0]
        if len(keyframe_idxs) == 2:
            last_stream = keyframe_idxs[-1]
        else:
            last_stream = keyframe_idxs[2]
        start = max(0, n - window_size)
        # 当前块对应的 keyframe video_idxs（最后 current_size 个）
        current_start = max(0, n - current_size)
        current_indices = keyframe_video_idxs[current_start:]
        # 窗口内流式索引范围：取与窗口内关键帧对应的首尾
        stream_start = keyframe_idxs[start] if start < len(keyframe_idxs) else first_stream
        stream_end = keyframe_idxs[-1]
        all_indices = list(range(stream_start, stream_end + 1))
        return all_indices, current_indices
    start = max(0, n - window_size)
    all_indices = keyframe_video_idxs[start:n]
    current_start = max(0, len(all_indices) - current_size)
    current_indices = all_indices[current_start:]
    return all_indices, current_indices


# ============================================================================
# 时序融合与全局先验
# ============================================================================

def temporal_fuse_p_dyn(
    p_dyn_list: List[torch.Tensor],
    flows_fwd: List[torch.Tensor],
    flows_bwd: List[torch.Tensor],
    cons_fwd: List[torch.Tensor],
    cons_bwd: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    时序融合：对每帧 t，将前一帧、后一帧的 p_dyn 按光流 warp 到 t 上，与当前帧 p_dyn 加权平均。
    """
    T = len(p_dyn_list)
    fused: List[torch.Tensor] = []
    for t in range(T):
        p = p_dyn_list[t]
        num = p.clone()
        den = torch.ones_like(p)
        if t > 0:
            p_prev = p_dyn_list[t - 1]
            w_prev = cons_fwd[t - 1] if t - 1 < len(cons_fwd) else torch.ones_like(p_prev)
            num = num + w_prev * warp_with_flow(p_prev, flows_fwd[t - 1])
            den = den + w_prev
        if t < T - 1:
            p_next = p_dyn_list[t + 1]
            w_next = cons_bwd[t] if t < len(cons_bwd) else torch.ones_like(p_next)
            num = num + w_next * warp_with_flow(p_next, flows_bwd[t])
            den = den + w_next
        p_fused = (num / (den + 1e-6)).clamp(0.0, 1.0)
        fused.append(p_fused)
    return fused

def temporal_fuse_p_dyn_refined(p_dyn_list, flows_fwd, flows_bwd, cons_fwd, cons_bwd):
    T = len(p_dyn_list)
    fused = []
    
    # 超参数
    curr_weight = 2.0  # 增加当前帧的权重话语权
    diff_thresh = 0.5  # 差异门控
    
    for t in range(T):
        p_curr = p_dyn_list[t]
        num = p_curr * curr_weight
        den = torch.ones_like(p_curr) * curr_weight
        
        # 处理前一帧
        if t > 0:
            warped_prev = warp_with_flow(p_dyn_list[t-1], flows_fwd[t-1])
            # 门控：如果 warped 值与当前观测值偏差太大，降低其权重
            gate_prev = (torch.abs(warped_prev - p_curr) < diff_thresh).float()
            w = cons_fwd[t-1] * gate_prev
            num += w * warped_prev
            den += w
            
        # 处理后一帧（逻辑同上）
        if t < T - 1:
            warped_next = warp_with_flow(p_dyn_list[t+1], flows_bwd[t])
            gate_next = (torch.abs(warped_next - p_curr) < diff_thresh).float()
            w = cons_bwd[t] * gate_next
            num += w * warped_next
            den += w
            
        fused.append((num / (den + 1e-6)).clamp(0.0, 1.0))
    return fused


def compute_global_prior(
    p_fused: List[torch.Tensor],
    flows_fwd: List[torch.Tensor],
    flows_bwd: List[torch.Tensor],
    cons_fwd: List[torch.Tensor],
    cons_bwd: List[torch.Tensor],
    ref_t: int,
    static_thr: float = GLOBAL_STATIC_THR,
    dyn_thr: float = GLOBAL_DYNAMIC_THR,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    全局先验：将所有帧的 p_fused 按光流 warp 到参考帧 ref_t 上投票，得到 p_ref（动态概率）、
    static_ref（静态先验）、dyn_ref（动态先验），均在参考帧坐标系下。
    """
    device = p_fused[0].device
    H, W = p_fused[0].shape
    count_dyn = torch.zeros(H, W, device=device, dtype=torch.float32)
    count_seen = torch.zeros(H, W, device=device, dtype=torch.float32)

    def accumulate(t: int, ref: int, p: torch.Tensor) -> None:
        nonlocal count_dyn, count_seen
        if t == ref:
            count_dyn = count_dyn + p
            count_seen = count_seen + 1
            return
        if t < ref:
            x, w = p, torch.ones_like(p)
            for k in range(t, ref):
                if k < len(flows_fwd):
                    x = warp_with_flow(x, flows_fwd[k])
                    w = warp_with_flow(w * cons_fwd[k], flows_fwd[k])
            count_dyn = count_dyn + x * w
            count_seen = count_seen + w
        else:
            x, w = p, torch.ones_like(p)
            for k in range(t - 1, ref - 1, -1):
                if k < len(flows_bwd):
                    x = warp_with_flow(x, flows_bwd[k])
                    w = warp_with_flow(w * cons_bwd[k], flows_bwd[k])
            count_dyn = count_dyn + x * w
            count_seen = count_seen + w

    for t in range(len(p_fused)):
        accumulate(t, ref_t, p_fused[t])

    p_ref = (count_dyn / (count_seen + 1e-6)).clamp(0.0, 1.0)
    static_ref = p_ref < static_thr
    dyn_ref = p_ref > dyn_thr
    return p_ref, static_ref, dyn_ref


def warp_global_prior_to_frame(
    p_ref: torch.Tensor,
    ref_t: int,
    t: int,
    flows_fwd: List[torch.Tensor],
    flows_bwd: List[torch.Tensor],
) -> torch.Tensor:
    """将参考帧上的 p_ref warp 到第 t 帧坐标系。"""
    if t == ref_t:
        return p_ref
    if t < ref_t:
        x = p_ref
        for k in range(ref_t - 1, t - 1, -1):
            if k < len(flows_bwd):
                x = warp_with_flow(x, flows_bwd[k])
        return x
    else:
        x = p_ref
        for k in range(ref_t, t):
            if k < len(flows_fwd):
                x = warp_with_flow(x, flows_fwd[k])
        return x


def get_yolo_dynamic_mask(
    frame_reader: Any,
    video_idx: int,
    image: torch.Tensor,
    device: torch.device,
    seg_chair: bool = False,
) -> Optional[torch.Tensor]:
    """
    用 dataset 的 YOLO 得到当前帧上的动态区域二值 mask（True=动态）。
    image: [C,H,W] 或 [1,C,H,W]，取值 [0,1]。
    """
    yolo = getattr(frame_reader, "yolo_model", None)
    if yolo is None:
        return None
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image_batch = image.clamp(0.0, 1.0).to(device)
    h, w = image_batch.shape[2], image_batch.shape[3]
    combined = torch.zeros((h, w), device=device, dtype=torch.bool)
    try:
        results = yolo.predict(
            source=image_batch,
            classes=[0],
            save=False,
            stream=False,
            show=False,
            verbose=False,
            device=device,
        )
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    m = mask.to(torch.bool)
                    if m.shape[0] != h or m.shape[1] != w:
                        m = torch.nn.functional.interpolate(
                            m.unsqueeze(0).unsqueeze(0).float(),
                            size=(h, w),
                            mode="nearest",
                        ).squeeze().bool()
                    combined = combined | m.to(device)
        if seg_chair:
            results = yolo.predict(
                source=image_batch,
                classes=[56],
                save=False,
                stream=False,
                show=False,
                verbose=False,
                device=device,
            )
            for result in results:
                if result.masks is not None:
                    for mask in result.masks.data:
                        m = mask.to(torch.bool)
                        if m.shape[0] != h or m.shape[1] != w:
                            m = torch.nn.functional.interpolate(
                                m.unsqueeze(0).unsqueeze(0).float(),
                                size=(h, w),
                                mode="nearest",
                            ).squeeze().bool()
                        combined = combined | m.to(device)
    except Exception:
        return None
    return combined


# ============================================================================
# 后端接口函数
# ============================================================================

def maybe_update_motion_masks(
    cameras: Dict[int, Any],
    keyframe_video_idxs: List[int],
    frame_reader: Any,
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
    interval: int = DETECT_INTERVAL,
    window_size: int = WINDOW_SIZE,
    current_block_size: int = CURRENT_BLOCK_SIZE,
    skip_if_existing: bool = False,
    keyframe_idxs: Optional[List[int]] = None,
) -> bool:
    """
    每个关键帧触发：若传入 keyframe_idxs（流式帧索引），则使用首尾之间的所有帧参与光流/时序/全局先验；
    否则仅用关键帧。对当前区间内关键帧相机用 p_fused 写回 motion_mask（True=静态）。
    skip_if_existing=True 时仅对尚未有有效 motion_mask 的相机写入；False 时一律覆盖更新。
    """
    config = config or {}
    enable = config.get("enable_motion_mask_update", True)
    if not enable:
        return False
    n = len(keyframe_video_idxs)
    if n < 2:
        return False

    # 若提供 keyframe_idxs 则使用首尾之间的所有帧；否则仅关键帧
    all_indices, current_indices = build_window_indices(
        keyframe_video_idxs,
        window_size=window_size,
        current_size=current_block_size,
        keyframe_idxs=keyframe_idxs,
    )
    if len(all_indices) < 2:
        return False

    use_all_frames = keyframe_idxs is not None and len(keyframe_idxs) == n
    if use_all_frames:
        # all_indices 为流式帧索引 [first_stream, ..., last_stream]，为每帧构造 viewpoint
        stream_to_video = {keyframe_idxs[i]: keyframe_video_idxs[i] for i in range(n)}
        ref_cam = next((cameras[vi] for vi in cameras if getattr(cameras[vi], "original_image", None) is not None), None)
        if ref_cam is None:
            return False
        viewpoints = []
        valid_indices = []
        for s in all_indices:
            if s in stream_to_video:
                vi = stream_to_video[s]
                cam = cameras.get(vi)
                if cam is not None and getattr(cam, "original_image", None) is not None:
                    viewpoints.append(cam)
                    valid_indices.append(s)
            else:
                vp = _make_minimal_viewpoint(frame_reader, s, device, ref_cam)
                if vp is not None:
                    viewpoints.append(vp)
                    valid_indices.append(s)
        all_indices = valid_indices
    else:
        viewpoints = [cameras[vi] for vi in all_indices if cameras.get(vi) is not None and getattr(cameras[vi], "original_image", None) is not None]
        all_indices = [vi for vi in all_indices if cameras.get(vi) is not None and getattr(cameras[vi], "original_image", None) is not None]

    if len(viewpoints) < 2:
        return False

    static_thr = config.get("global_static_thr", GLOBAL_STATIC_THR)
    dyn_thr = config.get("global_dynamic_thr", GLOBAL_DYNAMIC_THR)
    save_dir = config.get("save_debug_mask_dir")
    sam_refine_fn = config.get("sam_refine_fn") or _get_sam_refine_fn()
    seg_chair = config.get("seg_chair", False)

    with torch.no_grad():
        p_single, p_fused, p_ref, static_ref, dyn_ref = dynamic_segmentation_pipeline(
            viewpoints,
            device,
            local_k=config.get("local_k", FLOW_INCONS_LOCAL_K),
            incons_percentile=config.get("incons_percentile", DINO_FLOW_SEED_PERCENTILE),
            min_blob_area_ratio=config.get("min_blob_area_ratio", MIN_BLOB_AREA_RATIO),
            static_thresh=static_thr,
            dynamic_thresh=dyn_thr,
            save_dir=save_dir,
            video_idxs=all_indices,
            frame_reader=frame_reader,
            sam_refine_fn=sam_refine_fn,
            seg_chair=seg_chair,
        )

    if len(p_fused) < 1:
        return False

    # 写回：仅对关键帧 camera 写 motion_mask；若为「所有帧」模式则用 keyframe_idxs 将 video_idx 映到流式索引
    updated = 0
    for vi in current_indices:
        cam = cameras.get(vi)
        if cam is None or cam.original_image is None:
            continue
        if use_all_frames and keyframe_idxs is not None:
            try:
                pos = keyframe_video_idxs.index(vi)
                stream_idx = keyframe_idxs[pos] if pos < len(keyframe_idxs) else None
            except ValueError:
                stream_idx = None
            if stream_idx is None or stream_idx not in all_indices:
                continue
            idx_in_window = all_indices.index(stream_idx)
        else:
            if vi not in all_indices:
                continue
            try:
                idx_in_window = all_indices.index(vi)
            except ValueError:
                continue
        if idx_in_window >= len(p_fused):
            continue
        if skip_if_existing:
            m = getattr(cam, "motion_mask", None)
            if m is not None and m.dim() >= 2 and not m.all():
                continue
        # motion_mask True=静态：p_fused < static_thr 判定，并写回 camera
        p_fused_t = p_fused[idx_in_window]
        motion_mask = (p_fused_t < static_thr).to(torch.bool)
        cam.motion_mask = motion_mask
        # 按实际帧 id（流式索引）保存，而非关键帧编号
        if keyframe_idxs is not None:
            try:
                pos = keyframe_video_idxs.index(vi)
                save_idx = keyframe_idxs[pos] if pos < len(keyframe_idxs) else vi
            except ValueError:
                save_idx = vi
        else:
            save_idx = vi
        _save_motion_mask_by_idx(motion_mask, save_idx, config)
        updated += 1

    return updated > 0


def _save_motion_mask_by_idx(motion_mask: torch.Tensor, save_idx: int, config: Dict[str, Any]) -> None:
    """将 motion_mask 按 save_idx 保存到 config['save_dir']/motion_mask/ 下。"""
    base_dir = config.get("save_dir")
    if not base_dir:
        return
    out_dir = os.path.join(base_dir, "motion_mask")
    os.makedirs(out_dir, exist_ok=True)
    if motion_mask.dim() >= 2:
        mask_np = motion_mask.detach().cpu().numpy().astype(np.uint8)
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        # 保存为 PNG：255=静态(True)，0=动态(False)，便于查看
        mask_uint8 = np.where(mask_np > 0, 255, 0).astype(np.uint8)
        path_png = os.path.join(out_dir, f"{save_idx:06d}.png")
        cv2.imwrite(path_png, mask_uint8)
        path_npy = os.path.join(out_dir, f"{save_idx:06d}.npy")
        np.save(path_npy, mask_np.astype(bool))

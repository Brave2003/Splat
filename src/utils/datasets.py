# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from types import SimpleNamespace
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov
import torchvision.transforms as transforms
from ultralytics import YOLO
from utils.sam2_bridge import sam2_refine_dynamic_mask
#from segment_anything import sam_model_registry
def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y

def load_mono_depth(idx,path):
    # omnidata depth
    mono_depth_path = f"{path}/mono_priors/depths/{idx:05d}.npy"
    mono_depth = np.load(mono_depth_path)
    mono_depth_tensor = torch.from_numpy(mono_depth)
    
    return mono_depth_tensor  


def get_dataset(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'):
        # 调用父类构造函数
        super(BaseDataset, self).__init__()

        # 设置数据集名称和设备（默认为cuda:0）
        self.name = cfg['dataset']
        self.device = device

        # 获取深度图像缩放比例
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        # 初始化一些未定义的参数
        self.n_img = -1  # 图片数量，默认为-1
        self.depth_paths = None  # 深度图像路径
        self.color_paths = None  # 彩色图像路径
        self.poses = None  # 相机姿态
        self.normal_paths=None
        self.mask_paths=None
        self.mono_paths = None
        self.static_paths=None
        self.image_timestamps = None  # 图片时间戳

        # 从配置文件中获取相机内参的高度、宽度、焦距、光心位置
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        # 保存原始的内参
        self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig = self.fx, self.fy, self.cx, self.cy

        # 获取输出图像的尺寸
        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']

        # 获取图像边缘的尺寸（可能用于后处理或者填充）
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

        # 计算加上边缘后输出图像的尺寸
        self.H_out_with_edge, self.W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2

        # 将相机内参转为torch张量
        self.intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()

        # 根据输出图像的尺寸调整相机内参
        self.intrinsic[0] *= self.W_out_with_edge / self.W
        self.intrinsic[1] *= self.H_out_with_edge / self.H
        self.intrinsic[2] *= self.W_out_with_edge / self.W
        self.intrinsic[3] *= self.H_out_with_edge / self.H

        # 调整光心位置，使其适应边缘
        self.intrinsic[2] -= self.W_edge
        self.intrinsic[3] -= self.H_edge
        self.yolo_model=None
        # 更新相机内参的值
        self.fx = self.intrinsic[0].item()
        self.fy = self.intrinsic[1].item()
        self.cx = self.intrinsic[2].item()
        self.cy = self.intrinsic[3].item()

        self.fovx = focal2fov(self.fx, self.W_out)
        self.fovy = focal2fov(self.fy, self.H_out)

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None

        # retrieve input folder as temporary folder
        self.input_folder = os.path.join(cfg['data']['dataset_root'], cfg['data']['input_folder'])
        self.cfg = cfg
        self.motion_mask_dir = os.path.join(cfg["data"]["output"], cfg["scene"], "motion_mask")
        self.enable_dynamic_mask_detection = cfg.get(
            "enable_dynamic_mask_detection",
            cfg.get("data", {}).get("enable_dynamic_mask_detection", False),
        )
        self.motion_mask_255_is_static = cfg.get(
            "motion_mask_255_is_static",
            cfg.get("data", {}).get("motion_mask_255_is_static", True),
        )
        data_cfg = cfg.get("data", {})
        self.use_dsine_normal = cfg.get(
            "use_dsine_normal",
            data_cfg.get("use_dsine_normal", False),
        )
        # print("[DEBUG] self.use_dsine_normal", self.use_dsine_normal)
        self.dsine_pretrained = cfg.get(
            "dsine_pretrained",
            data_cfg.get("dsine_pretrained", "pretrained/dsine.pt"),
        )
        self.dsine_architecture = cfg.get(
            "dsine_architecture",
            data_cfg.get("dsine_architecture", "v02"),
        )
        self.dsine_num_iter_test = int(
            cfg.get(
                "dsine_num_iter_test",
                data_cfg.get("dsine_num_iter_test", 5),
            )
        )
        self.dsine_model = None
        self.dsine_ready = False
        self.dsine_disabled_reason = None
        self.dynamic_objects = 0
        self.seg_chair = True if "seg_chair" in cfg["meshing"].keys() else False
        self.seg_ballon=True if "seg_ballon" in cfg["meshing"].keys() else False

    # def yolo_model(self):
    #     if self._yolo_model is not None:
    #         #raise AttributeError("YOLO模型未加载，请先调用load_yolo()")
    #       return self._yolo_model

    def load_yolo(self, model_path=None):
        """ 正确的模型加载方法（移除@property装饰器） """
        if model_path is None:
            model_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'pretrained', 'yolo11l-seg.pt'))

        # 检查设备可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法加载YOLO模型")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO模型文件不存在: {model_path}")

        # 加载并分配设备
        self.yolo_model = YOLO(model_path).to(self.device)

    def _load_dsine(self):
        if self.dsine_ready:
            return True
        if self.dsine_disabled_reason is not None:
            return False

        ckpt_path = self.dsine_pretrained
        if not os.path.isabs(ckpt_path):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            ckpt_path = os.path.join(project_root, ckpt_path)
        if not os.path.isfile(ckpt_path):
            self.dsine_disabled_reason = f"DSINE checkpoint not found: {ckpt_path}"
            print(self.dsine_disabled_reason)
            return False

        try:
            if self.dsine_architecture == "v02_kappa":
                from thirdparty.DSINE.models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
            elif self.dsine_architecture == "v02":
                from thirdparty.DSINE.models.dsine.v02 import DSINE_v02 as DSINE
            elif self.dsine_architecture == "v01":
                from thirdparty.DSINE.models.dsine.v01 import DSINE_v01 as DSINE
            elif self.dsine_architecture == "v00":
                from thirdparty.DSINE.models.dsine.v00 import DSINE_v00 as DSINE
            else:
                raise ValueError(f"Unsupported DSINE architecture: {self.dsine_architecture}")

            args = SimpleNamespace(
                NNET_architecture=self.dsine_architecture,
                NNET_output_dim=4 if self.dsine_architecture == "v02_kappa" else 3,
                NNET_output_type="R",
                NNET_feature_dim=64,
                NNET_hidden_dim=64,
                NNET_encoder_B=5,
                NNET_decoder_NF=2048,
                NNET_decoder_BN=False,
                NNET_decoder_down=8,
                NNET_learned_upsampling=True,
                NRN_prop_ps=5,
                NRN_num_iter_train=5,
                NRN_num_iter_test=self.dsine_num_iter_test,
                NRN_ray_relu=True,
            )

            model = DSINE(args).to(self.device)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            cleaned = {}
            for key, value in state_dict.items():
                cleaned[key.replace("module.", "")] = value
            model.load_state_dict(cleaned, strict=True)
            self.dsine_model = model.eval()
            self.dsine_ready = True
            return True
        except Exception as exc:
            self.dsine_disabled_reason = f"Failed to init DSINE: {exc}"
            print(self.dsine_disabled_reason)
            self.dsine_model = None
            self.dsine_ready = False
            return False

    @staticmethod
    def _get_pad_lrtb(h, w):
        if w % 32 == 0:
            l, r = 0, 0
        else:
            new_w = 32 * ((w // 32) + 1)
            l = (new_w - w) // 2
            r = (new_w - w) - l

        if h % 32 == 0:
            t, b = 0, 0
        else:
            new_h = 32 * ((h // 32) + 1)
            t = (new_h - h) // 2
            b = (new_h - h) - t
        return l, r, t, b

    @torch.no_grad()
    def _predict_dsine_normal(self, rgb_1x3xhxw):
        if not self._load_dsine():
            return None

        try:
            inp = rgb_1x3xhxw.to(self.device)
            _, _, h, w = inp.shape
            l, r, t, b = self._get_pad_lrtb(h, w)
            if l + r + t + b > 0:
                inp = F.pad(inp, (l, r, t, b), mode="constant", value=0.0)

            mean = torch.tensor([0.485, 0.456, 0.406], device=inp.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=inp.device).view(1, 3, 1, 1)
            inp = (inp - mean) / std

            intrins = torch.tensor(
                [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=inp.device,
            ).unsqueeze(0)
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t

            pred_norm = self.dsine_model(inp, intrins=intrins, mode="test")[-1]
            pred_norm = pred_norm[:, :, t:t + h, l:l + w]
            pred_norm = -pred_norm
            return pred_norm.detach().cpu().float()
        except Exception as exc:
            print(f"Failed DSINE normal inference: {exc}")
            return None
    def __len__(self):
        return self.n_img

    def depthloader(self, index, depth_paths, depth_scale):
        if depth_paths is None:
            return None
        depth_path = depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / depth_scale

        return depth_data

    # 修改后的 datasets.py 部分代码
    # def depthloader(self,index, depth_paths, depth_scale):
    #     # 检查索引有效性
    #     if index >= len(depth_paths):
    #         raise IndexError(f"索引 {index} 越界，深度图列表长度：{len(depth_paths)}")
    #
    #     depth_path = depth_paths[index]
    #
    #     # 检查文件存在性
    #     if not os.path.exists(depth_path):
    #         raise FileNotFoundError(f"深度图文件不存在：{depth_path}")
    #
    #     # 读取文件
    #     try:
    #         if depth_path.endswith(".png"):
    #             depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    #         elif depth_path.endswith(".exr"):
    #             depth_data = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #         else:
    #             raise ValueError(f"不支持的深度图格式：{depth_path}")
    #     except Exception as e:
    #         raise RuntimeError(f"读取深度图失败：{str(e)}")
    #
    #     # 检查数据有效性
    #     if depth_data is None:
    #         raise ValueError(f"无法解析深度图：{depth_path}")
    #     if not np.issubdtype(depth_data.dtype, np.number):
    #         raise TypeError(f"深度图数据非数值类型：{depth_path}")
    #
    #     # 检查缩放因子
    #     if depth_scale <= 0 or not isinstance(depth_scale, (int, float)):
    #         raise ValueError(f"无效的深度缩放因子：{depth_scale}")
    #
    #     # 转换和缩放
    #     depth_data = depth_data.astype(np.float32) / depth_scale
    #     return depth_data

    def normalloader(self, index, normal_paths, normal_scale=1.0):
        """
        读取法向量图的函数

        参数:
            index: 要读取的法向量图索引
            normal_paths: 法向量图路径列表
            normal_scale: 法向量值缩放因子(默认为1.0)

        返回:
            normal_data: 法向量数据(torch.Tensor, shape: [H,W,3] 或 [3,H,W])
        """
        # 检查索引有效性
        if index >= len(normal_paths):
            raise IndexError(f"索引 {index} 越界，法向量图列表长度：{len(normal_paths)}")

        normal_path = normal_paths[index]

        # 检查文件存在性
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"法向量图文件不存在：{normal_path}")

        # 读取文件
        try:
            if normal_path.endswith(".png"):
                # PNG格式通常将法向量存储在RGB通道中，值范围[0,255]
                normal_data = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
                if normal_data is not None and len(normal_data.shape) == 3:
                    normal_data = cv2.cvtColor(normal_data, cv2.COLOR_BGR2RGB)  # 转换为RGB顺序
            elif normal_path.endswith(".exr"):
                # EXR格式可以直接存储浮点法向量
                normal_data = cv2.imread(normal_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            else:
                raise ValueError(f"不支持的法向量图格式：{normal_path}")
        except Exception as e:
            raise RuntimeError(f"读取法向量图失败：{str(e)}")

        # 检查数据有效性
        if normal_data is None:
            raise ValueError(f"无法解析法向量图：{normal_path}")
        if not np.issubdtype(normal_data.dtype, np.number):
            raise TypeError(f"法向量图数据非数值类型：{normal_path}")

        # 检查缩放因子
        if normal_scale <= 0 or not isinstance(normal_scale, (int, float)):
            raise ValueError(f"无效的法向量缩放因子：{normal_scale}")

        # 转换和缩放
        if normal_path.endswith(".png"):
            # PNG格式的法向量通常需要从[0,255]映射到[-1,1]
            normal_data = normal_data.astype(np.float32) / 255.0 * 2.0 - 1.0
        else:
            # EXR或其他格式直接转换为浮点数
            normal_data = normal_data.astype(np.float32)

        # 应用缩放因子
        normal_data = normal_data * normal_scale

        # 确保法向量是单位向量(可选)
        # norm = np.linalg.norm(normal_data, axis=-1, keepdims=True)
        # normal_data = normal_data / (norm + 1e-6)

        # 转换为PyTorch张量并调整维度顺序
        normal_data = torch.from_numpy(normal_data).float()
        if len(normal_data.shape) == 3:  # 如果有通道维度
            normal_data = normal_data.permute(2, 0, 1)  # 从HWC转为CHW

        return normal_data
    def get_color(self,index):
        # not used now
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        color_data = cv2.resize(color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
        return color_data

    def get_intrinsic(self):
        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        intrinsic = torch.as_tensor([self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H   
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge   
        return intrinsic 

    def __getitem__(self, index):
        mono_data = None
        static_msk = None
        normal_input = None
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        
        outsize = (self.H_out_with_edge, self.W_out_with_edge)

        color_data = cv2.resize(color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        # color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)
        if self.yolo_model is  None:
            self.load_yolo()
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]
        if self.mono_paths and index < len(self.mono_paths):
            mono_data=self.mono_paths[index]
            mono_data= np.load(mono_data)
            mono_data = mono_data.astype(np.float32)
            if mono_data is not None:
                mono_data = torch.from_numpy(mono_data).float()#/ self.png_depth_scale
                mono_data = F.interpolate(
                    mono_data[None, None], outsize, mode='nearest')[0, 0]

                if self.W_edge > 0:
                    edge = self.W_edge
                    mono_data = mono_data[:, edge:-edge]
                if self.H_edge > 0:
                    edge = self.H_edge
                    mono_data = mono_data[edge:-edge, :]
        depth_data_fullsize = self.depthloader(index,self.depth_paths,self.png_depth_scale)
        if depth_data_fullsize is not None:
            depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
            depth_data = F.interpolate(
                depth_data_fullsize[None, None], outsize, mode='nearest')[0, 0]

        if self.static_paths and index < len(self.static_paths):
            static_data = self.static_paths[index]
            static_msk = None
            static_data = np.load(static_data)
            dist_flow = np.linalg.norm(static_data["flow"], ord=2, axis=-1)
            dist_flow = cv2.resize(dist_flow, (self.W_out_with_edge, self.H_out_with_edge))
            if static_msk is None:
                static_msk = np.ones_like(dist_flow)

            static_msk = np.logical_and(static_msk, dist_flow < 0.8)
            if self.W_edge > 0:
                edge = self.W_edge
                static_msk = static_msk[:, edge:-edge]
            if self.H_edge > 0:
                edge = self.H_edge
                static_msk = static_msk[edge:-edge, :]
        else :
            static_msk=None
        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            depth_data = depth_data[:, edge:-edge]
        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            depth_data = depth_data[edge:-edge, :]

        mask_input = torch.zeros((self.H_out, self.W_out), device=self.device, dtype=torch.bool)
        if self.mask_paths and index < len(self.mask_paths):
            try:
                mask_img = Image.open(self.mask_paths[index]).convert("L")  # 加载灰度掩码
                mask_tensor = transforms.ToTensor()(mask_img)  # [1, H_orig, W_orig]
                mask = (mask_tensor > 0.01).to(torch.bool).squeeze(0)  # [H_orig, W_orig]

                # 调整掩码大小以匹配调整后的图像尺寸
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  # 添加批次和通道维度
                    size=outsize,
                    mode='nearest'
                ).squeeze().to(torch.bool)  # [H_out, W_out]
                if self.W_edge > 0:
                    edge = self.W_edge
                    mask_resized = mask_resized[:, edge:-edge]
                if self.H_edge > 0:
                    edge = self.H_edge
                    mask_resized = mask_resized[edge:-edge, :]
                # 添加膨胀操作
                mask_np = mask_resized.cpu().numpy().astype(np.uint8) * 255
                kernel_size = 7  # 与第二段代码相同的核大小
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                dilated_np = cv2.dilate(mask_np, kernel, iterations=1)

                # 转回张量
                dilated_mask = torch.from_numpy(dilated_np > 0).to(device=self.device, dtype=torch.bool)
                mask_input = dilated_mask

            except Exception as e:
                print(f"预生成掩码加载失败: {e}")
        
        if self.use_dsine_normal:
            normal_input = self._predict_dsine_normal(color_data)
            if normal_input is not None:
                normal_input = normal_input.to(self.device)
            normal_input[:, [0, 2], :, :] = normal_input[:, [2, 0], :, :]
            normal_input = (normal_input)/255 * 2.0 - 1.0

        if normal_input is None and self.normal_paths and index < len(self.normal_paths):
            try:
                normal_path = self.normal_paths[index]
                normal_data= np.load(normal_path)
                normal_data = cv2.cvtColor(normal_data, cv2.COLOR_BGR2RGB)
                normal_data = cv2.resize(normal_data, (self.W_out_with_edge, self.H_out_with_edge))
                normal_data= torch.from_numpy(normal_data).float().permute(2, 0, 1) / 255.0
                normal_data=normal_data.unsqueeze(dim=0)
                if self.W_edge > 0:
                    edge = self.W_edge
                    normal_data = normal_data[:, :, :, edge:-edge]

                if self.H_edge > 0:
                    edge = self.H_edge
                    normal_data = normal_data[:,  :,edge:-edge, :]
                #normal_input  = torch.from_numpy(normal_data).float().permute(2, 0, 1) / 255.0

                # 将normal值从[0,1]映射到[-1,1]
                normal_input  = normal_data * 2.0 - 1.0
            except Exception as e:
                print(f"Failed to load normal map: {e}")

        mask_from_disk = None
        if self.motion_mask_dir:
            disk_mask_path = os.path.join(self.motion_mask_dir, f"{index:06d}.png")
            if os.path.isfile(disk_mask_path):
                disk_mask = cv2.imread(disk_mask_path, cv2.IMREAD_GRAYSCALE)
                if disk_mask is not None:
                    if self.motion_mask_255_is_static:
                        disk_mask = (disk_mask > 127).astype(np.float32)
                    else:
                        disk_mask = (disk_mask <= 127).astype(np.float32)
                    mask_from_disk = torch.from_numpy(disk_mask).unsqueeze(0).unsqueeze(0)
                    mask_from_disk = F.interpolate(
                        mask_from_disk,
                        size=(self.H_out, self.W_out),
                        mode="nearest",
                    ).squeeze().bool().to(self.device)

        if mask_from_disk is not None:
            final_mask = mask_from_disk
        elif self.enable_dynamic_mask_detection:
            final_mask = torch.logical_not(mask_input)
        else:
            combined_mask = torch.zeros((self.H_out, self.W_out), device=self.device, dtype=torch.bool)
            if self.yolo_model is not None:
                results = self.yolo_model.track(source=color_data, persist=True,classes=[0], save=False, stream=False,
                                                  show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                            kernel_size = 7
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                            dilated_np = cv2.dilate(mask_np, kernel, iterations=1)
                            dilated_mask = torch.from_numpy(dilated_np > 0).to(device=mask.device, dtype=torch.bool)
                            combined_mask |= dilated_mask
            if self.yolo_model is not None and self.seg_chair:
                results = self.yolo_model.predict(source=color_data, classes=[56], save=False, stream=False,
                                                  show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                            kernel_size = 7
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                            dilated_np = cv2.dilate(mask_np, kernel, iterations=1)
                            dilated_mask = torch.from_numpy(dilated_np > 0).to(device=mask.device, dtype=torch.bool)
                            combined_mask |= dilated_mask
            if self.yolo_model is not None and self.seg_ballon:
                results = self.yolo_model.predict(source=color_data, classes=[1], save=False, stream=False,
                                                  show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                            kernel_size = 7
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                            dilated_np = cv2.dilate(mask_np, kernel, iterations=1)
                            dilated_mask = torch.from_numpy(dilated_np > 0).to(device=mask.device, dtype=torch.bool)
                            combined_mask |= dilated_mask
            if bool(combined_mask.any().item()):
                rgb_np = (color_data[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                refined_dynamic_mask = sam2_refine_dynamic_mask(
                    rgb_np=rgb_np,
                    pos_mask=combined_mask.detach().cpu().numpy().astype(np.uint8),
                    neg_mask=np.zeros((self.H_out, self.W_out), dtype=np.uint8),
                )
                if refined_dynamic_mask is not None:
                    combined_mask = torch.from_numpy(refined_dynamic_mask).to(device=self.device, dtype=torch.bool)
            final_mask = torch.logical_not(combined_mask | mask_input)
        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float() #torch.from_numpy(np.linalg.inv(self.poses[0]) @ self.poses[index]).float()
        else:
            pose = None
        #print(f"Raw pose at index {index}:{self.poses[index]}")
        color_data_fullsize = cv2.cvtColor(color_data_fullsize,cv2.COLOR_BGR2RGB)
        color_data_fullsize = color_data_fullsize / 255.
        color_data_fullsize = torch.from_numpy(color_data_fullsize)
        return index, color_data, depth_data, pose,final_mask,normal_input,mono_data,static_msk


class Replica(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(Replica, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)

        self.load_poses(f'{self.input_folder}/traj.txt')
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(ScanNet, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, device)
        self.color_paths, self.depth_paths, self.poses,self.mask_paths ,self.normal_paths,self.mono_paths,self.static_paths= self.loadtum(
            self.input_folder, frame_rate=32)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.mask_paths=self.mask_paths[:max_frames][::stride]
        self.normal_paths=self.normal_paths[:max_frames][::stride]
        self.mono_paths = self.mono_paths[:max_frames][::stride]
        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations
    def extract_number(self, file_path):
        """从文件名提取完整的Unix时间戳（如 '1548266542.97977.png'）"""
        import re
        basename = os.path.basename(file_path)  # 获取文件名（不含路径）
        match = re.search(r'^(\d+\.\d+)', basename)  # 匹配开头的浮点数
        if match:
            return float(match.group(1))
        return 0.0  # 默认值

    def _collect_optional_files(self, datapath, subdir, patterns, message):
        subdir_path = os.path.join(datapath, subdir)
        if not os.path.isdir(subdir_path):
            return None

        matched_paths = []
        for pattern in patterns:
            matched_paths.extend(
                glob.glob(os.path.join(self.input_folder, subdir, pattern), recursive=True)
            )

        matched_paths = sorted(set(matched_paths), key=self.extract_number)
        if len(matched_paths) == 0:
            return None

        return matched_paths

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')
        mask_paths = []
        self.mask_path = self._collect_optional_files(
            datapath, "render_mask", ["*.png", "**/*.png"], "Using mask image"
        )
        normal_paths = []
        self.normal_path = self._collect_optional_files(
            datapath, "normal", ["*.npy", "*.png", "**/*.npy", "**/*.png"], "Using normal maps"
        )
        mono_paths=[]
        self.mono_path = self._collect_optional_files(
            datapath, "depth_npy", ["*.npy", "**/*.npy"], "Using mono maps"
        )
        static_paths=[]
        self.static_path = self._collect_optional_files(
            datapath, "flow_RAFT1", ["*.npz", "**/*.npz"], "Using static flow maps"
        )
        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        max_image_index = max((assoc[0] for assoc in associations), default=-1)
        optional_sources = [
            ("mask_path", "mask"),
            ("normal_path", "normal"),
            ("mono_path", "mono"),
            ("static_path", "static"),
        ]
        for attr_name, label in optional_sources:
            attr_value = getattr(self, attr_name)
            if attr_value is not None and len(attr_value) <= max_image_index:
                print(
                    f"Skip {label} maps: only found {len(attr_value)} files for "
                    f"{max_image_index + 1} frames."
                )
                setattr(self, attr_name, None)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            # timestamp tx ty tz qx qy qz qw
            if self.mask_path is not None:
                mask_paths += [self.mask_path[i]]
            if self.normal_path is not None:
                normal_paths += [self.normal_path[i]]
            if self.mono_path is not None:
                mono_paths += [self.mono_path[i]]
            if self.static_path is not None:
                static_paths += [self.static_path[i]]
            # c2w = pose_vecs[k]  # 提取相机到世界坐标系的位姿向量
            # c2w = torch.from_numpy(c2w).float()  # 转为PyTorch张量

            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            poses += [c2w]

        return images, depths, poses,mask_paths,normal_paths,mono_paths,static_paths

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose



dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD,
}

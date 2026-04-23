# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
from datetime import datetime
import numpy as np
import torch
import lietorch
import droid_backends
import thirdparty.glorie_slam.geom.ba
from  thirdparty.glorie_slam.geom.ba import JDSA
from torch.multiprocessing import Value
from lietorch import SE3
from thirdparty.glorie_slam.modules.droid_net import cvx_upsample
import thirdparty.glorie_slam.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
from thirdparty.glorie_slam.warp.depth_warp import depth_warp_to_mask
from thirdparty.glorie_slam.pose_transform import quaternion_to_transform_noBatch
from thirdparty.glorie_slam.pgo_buffer import global_relative_posesim3_constraints
class DepthVideo:
    ''' 存储估计的位姿和深度图，
        在跟踪器和映射器之间共享数据 '''

    def __init__(self, cfg, printer):
        # 从配置中获取数据和场景路径，初始化输出路径
        self.cfg = cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        # 从配置中获取输出图像的高度和宽度
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        # 初始化关键帧计数器（使用共享内存）
        self.counter = Value('i', 0)  # 当前关键帧的计数器
        # 从配置中获取缓冲区大小、BA类型、单目阈值、设备等参数
        buffer = cfg['tracking']['buffer']
        self.BA_type = cfg['tracking']['backend']['BA_type']
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.down_scale = 8  # 图像下采样倍数
        self.ht_8 = self.ht // 8
        self.wd_8 = self.wd // 8
        ### 状态属性 ###
        # 初始化时间戳张量，记录缓存中的时间戳（使用共享内存）
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        # 初始化图像张量，记录缓存中的图像数据（使用共享内存）
        self.images = torch.zeros(buffer, 3, ht, wd, device=self.device, dtype=torch.uint8)
        # 初始化dirty标志，用于指示valid_depth_mask是否被计算/更新过
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()
        # 初始化npc_dirty标志，用于指示点云是否根据位姿和深度图发生了变形
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()
        # 初始化位姿张量，存储每个帧的位姿（7个参数：x, y, z, 四元数）
        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()
        # 初始化深度图张量，存储每个帧的深度图（下采样后）
        self.disps = torch.ones(buffer, ht // self.down_scale, wd // self.down_scale, device=self.device,
                                dtype=torch.float).share_memory_()
        # 初始化零值张量，用于深度图的后处理
        self.zeros = torch.zeros(buffer, ht // self.down_scale, wd // self.down_scale, device=self.device,
                                 dtype=torch.float).share_memory_()
        self.poses_sim3 = torch.zeros(buffer, 8, device="cuda", dtype=torch.float).share_memory_()
        # 初始化上采样后的深度图张量
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()

        # 初始化相机内参张量，存储每帧的相机内参（如焦距、主点等）
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.normals = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.float)
        self.poses_gt = torch.zeros(buffer, 4, 4, device=self.device, dtype=torch.float).share_memory_()
        self.depths = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.disps_mono_up = torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        # 初始化单目深度图（下采样后的深度图）
        self.mono_disps = torch.zeros(buffer, ht // self.down_scale, wd // self.down_scale, device=self.device,
                                      dtype=torch.float).share_memory_()

        # 初始化深度尺度和偏移量
        self.depth_scale = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        self.mask = torch.zeros(buffer, ht // 8, wd // 8, device="cuda", dtype=torch.bool).share_memory_()
        self.mask_ori = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.bool).share_memory_()
        # 初始化有效深度掩码，标记每个像素是否有有效深度信息
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()

        # 初始化下采样后的有效深度掩码
        self.valid_depth_mask_small = torch.zeros(buffer, ht // self.down_scale, wd // self.down_scale,
                                                  device=self.device, dtype=torch.bool).share_memory_()
        #self.poses_gt = torch.zeros(buffer, 7, device="cuda", dtype=torch.float32).share_memory_()
        ### 特征属性 ###
        # 初始化特征图，存储缓存中的每个图像的特征
        self.fmaps = torch.zeros(buffer, 1, 128, ht // self.down_scale, wd // self.down_scale, dtype=torch.half,
                                 device=self.device).share_memory_()

        # 初始化网络输出，存储每个图像的网络输出特征
        self.nets = torch.zeros(buffer, 128, ht // self.down_scale, wd // self.down_scale, dtype=torch.half,
                                device=self.device).share_memory_()

        # 初始化输入数据，存储输入给网络的数据
        self.inps = torch.zeros(buffer, 128, ht // self.down_scale, wd // self.down_scale, dtype=torch.half,
                                device=self.device).share_memory_()

        # 初始化位姿为单位变换（即零平移和单位四元数）
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        #self.poses_sim3[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float, device="cuda")
        self.dscales = torch.ones(buffer, 2, 2, device='cuda', dtype=torch.float).share_memory_()
       # self.doffset = torch.zeros(buffer, 1, 1, device='cuda', dtype=torch.float).share_memory_()
        self.use_segmask = True
        self.use_depth_warp = True
        self.use_depth_mask = True
        # 初始化打印器，用于处理和打印信息
        self.printer = printer

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1
        # #self.counter.value = index + 1
        # index=self.counter.value
        # self.counter.value = index + 1
        self.timestamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            save_dir = os.path.join("output", "depth_out")
            # os.makedirs(save_dir, exist_ok=True)
            file_name = f"depth_{index:04d}.png"
            file_path = os.path.join(save_dir, file_name)
            mono_depth = item[4][self.down_scale//2-1::self.down_scale,
                                 self.down_scale//2-1::self.down_scale]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            self.depths[index] = item[4]
            self.disps_mono_up[index]=1/item[4]
            depth_image = self.depths[index].cpu().numpy()
            # print("depth shape",depth_image.shape)
            min_depth = depth_image.min()
            max_depth = depth_image.max()
            #
            # # 归一化到 0-255 范围内（8位图像）
            depth_normalized_8bit = (( depth_image- min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
            #
            # # 或者，归一化到 0-65535 范围内（16位图像）
            # # depth_normalized_16bit = ((depth_image - min_depth) / (max_depth - min_depth) * 65535).astype(np.uint16)
            #
            # # 将深度图保存为文件
            # cv2.imwrite(file_path, depth_normalized_8bit)  # 保存为 8 位图像
            # # cv2.imwrite(file_path, depth_normalized_16bit)  # 保存为 16 位图像
            #

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]
        if len(item) > 9:
            if item[9] is not None:
                if len(item[9].shape) > 2:
                    seg_mask = item[9][:, 3::8,3::8]
                else:
                    seg_mask = item[9][3::8,3::8]
                #print("seg_mask",seg_mask.shape)
                #output_dir = os.path.join("output", "ori masks", datetime.now().strftime("%Y%m%d-%H%M%S"))
                #os.makedirs(output_dir, exist_ok=True)
                #mask_np = seg_mask.cpu().numpy().astype(np.uint8) * 255
                #output_path = os.path.join(output_dir, f"mask_{index:04d}.png")
                #cv2.imwrite(output_path, mask_np)

                #print(f"warp掩码已保存至 {output_path}")
                self.mask[index] = seg_mask
                self.mask_ori[index] = item[9]
        if len(item) > 10:
            self.poses_gt[index] = item[10]
        if len(item) > 11:
            self.normals[index] = item[11]
    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    # def to_dict(self):
    #     return {
    #         "frames": self.frames,  # 假设 frames 是列表或可序列化对象
    #         "intrinsics": self.intrinsics,  # 内参矩阵
    #         "depth_scale": self.depth_scale
    #     }
    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean().item() * self.cfg['tracking']['scale_multiplier']
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.disps_up[:self.counter.value] /= s
            self.dscales[:self.counter.value] /= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj,sim3=False):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.Sim3(self.poses_sim3[None]) if sim3 else lietorch.SE3(self.poses[None])
        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    # def shift(self, ix, n=1):
    #     with self.get_lock():
    #         self.timestamp[ix+n:self.counter.value+n] = self.timestamp[ix:self.counter.value].clone()
    #         self.images[ix+n:self.counter.value+n] = self.images[ix:self.counter.value].clone()
    #         self.dirty[ix+n:self.counter.value+n] = self.dirty[ix:self.counter.value].clone()
    #         self.poses[ix+n:self.counter.value+n] = self.poses[ix:self.counter.value].clone()
    #         self.poses_sim3[ix+n:self.counter.value+n] = self.poses_sim3[ix:self.counter.value].clone()
    #         self.disps[ix+n:self.counter.value+n] = self.disps[ix:self.counter.value].clone()
    #         self.mono_disps[ix+n:self.counter.value+n] = self.mono_disps[ix:self.counter.value].clone()
    #         self.disps_up[ix+n:self.counter.value+n] = self.disps_up[ix:self.counter.value].clone()
    #         self.disps_mono_up[ix+n:self.counter.value+n] = self.disps_mono_up[ix:self.counter.value].clone()
    #         self.intrinsics[ix+n:self.counter.value+n] = self.intrinsics[ix:self.counter.value].clone()
    #         #self.normals[ix+n:self.counter.value+n] = self.normals[ix:self.counter.value].clone()
    #         self.fmaps[ix+n:self.counter.value+n] = self.fmaps[ix:self.counter.value].clone()
    #         self.nets[ix+n:self.counter.value+n] = self.nets[ix:self.counter.value].clone()
    #         self.inps[ix+n:self.counter.value+n] = self.inps[ix:self.counter.value].clone()
    #         self.mask[ix + n:self.counter.value + n] = self.mask[ix:self.counter.value].clone()
    #         self.mask_ori[ix + n:self.counter.value + n] = self.mask_ori[ix:self.counter.value].clone()
    #         self.counter.value += n
    def distance_covis(self, ii=None):
        """ frame distance metric based on covisibility """
        ii = torch.as_tensor(ii)
        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        poses = self.poses[:self.counter.value].clone()
        d = droid_backends.covis_distance(poses, self.disps, self.intrinsics[0], ii)
        d = d * (1. / self.disps[ii].median())
        return d
    def dspo(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False,
             opt_type="pose_depth",use_mono=False):
        """ 视差、尺度与姿态联合优化层（DSPO），详见论文公式推导
            Disparity, Scale and Pose Optimization (DSPO) layer,
            checked the paper (and supplementary) for detailed explanation

            参数说明：
            opt_type: "pose_depth" - 阶段1，联合优化相机姿态与视差图（对应论文公式16，类似DBA方法）
                      "depth_scale" - 阶段2，联合优化视差图、深度尺度和位移（对应论文公式17）
            两个阶段交替执行
        """

        with self.get_lock():  # 线程安全锁

            # 定义滑动窗口优化范围[t0, t1]
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1  # 自动计算最大时间窗口
            # output_dir = os.path.join("output", "try_masks")
            # os.makedirs(output_dir, exist_ok=True)
            initial_mem = torch.cuda.memory_allocated() / 1024 ** 3
            #print(f"初始显存占用: {initial_mem:.2f}GB")
            ###################### 阶段1：姿态与视差联合优化 ######################
            #print("enter dspo")
            #ii = ii.to(self.mask.device)
            if opt_type == "pose_depth":
                # 数据维度重整（BCHW格式）
                # --- 动态掩码生成 ---
                seg_masks, warp_mask = None, None
                if self.use_segmask:  # 语义掩码（YOLO动态检测）
                    seg_masks = self.mask[ii]  # 假设已预生成并存储
                    seg_masks = ~seg_masks  # True表示动态区域
                    #print(f"语义掩码显存: {seg_masks.element_size() * seg_masks.nelement() / 1024 ** 3:.2f}GB")
                    #print("seg_mask",seg_masks.shape)
                    # last_index = ii[-1].item()
                    # print("last_index",last_index)
                    # mask_np = seg_masks.cpu().numpy().astype(np.uint8) * 255  # True->255, False->0
                    # print("mask shape", mask_np.shape)
                    # # 保存图像（文件名按索引命名）
                    # output_path = os.path.join(output_dir, f"mask_{last_index:04d}.png")
                    # cv2.imwrite(output_path, mask_np)

                    #print(f"warp掩码已保存至 {output_path}")
                if self.use_depth_warp:  # 几何掩码（深度变形）
                    warp_mask = []
                    for id in range(len(ii)):
                        i, j = ii[id], jj[id]
                        cur_depth = self.depths[j][3::8, 3::8].unsqueeze(0)
                        last_depth = self.depths[i][3::8, 3::8].unsqueeze(0)
                        #print("pose i ",self.poses[i])
                        #print("pose j",self.poses[j])
                        cur_pose = quaternion_to_transform_noBatch(SE3(self.poses[j]).inv().data).unsqueeze(0)
                        last_pose = quaternion_to_transform_noBatch(SE3(self.poses[i]).inv().data).unsqueeze(0)
                        #print("cur_pose:", cur_pose)
                        #print("last_pose:", last_pose)
                        # print("cur_depth 最小值/最大值:", cur_depth.min(), cur_depth.max())
                        # print("last_depth 最小值/最大值:", last_depth.min(), last_depth.max())
                        # print("相机内参:", self.intrinsics[0])
                        depth_range = cur_depth.max() - cur_depth.min()
                        threshold = depth_range.item() * 0.04
                        mask_i = depth_warp_to_mask(cur_pose, last_pose, cur_depth.unsqueeze(-1),
                                                    last_depth.unsqueeze(-1), cur_depth, self.intrinsics[0],
                                                    self.ht_8, self.wd_8, threshold=threshold)
                        true_ratio = mask_i.float().mean().item()
                        #print(f"动态区域占比: {true_ratio * 100:.2f}%")
                        # print("id",id)
                        # print("i",i)
                        # print("j",j)
                        warp_mask.append(mask_i)
                        #print("mask shape",mask_np.shape)
                        # 保存图像（文件名按索引命名）
                        del  cur_depth, last_depth, cur_pose, last_pose
                        #print(f"warp掩码已保存至 {output_path}")
                    warp_mask = torch.cat(warp_mask, dim=0)

                # --- 动态区域权重抑制 ---
                if self.use_segmask or self.use_depth_warp:
                    final_mask =seg_masks#(seg_masks | warp_mask) if (seg_masks is not None and warp_mask is not None) else \
                         #(seg_masks if seg_masks is not None else warp_mask)
                    for i in range(final_mask.shape[0]):
                        frame = final_mask[i]
                        final_mask_np = frame.cpu().numpy().astype(np.uint8) * 255  # True->255, False->0
                        #print("final_mask_np shape", final_mask_np.shape)
                        # 保存图像（文件名按索引命名）
                        del final_mask_np
                        #print(f"final_mask掩码已保存至 {output_path}")
                    #weight = weight * (0.1 * final_mask + 0.9 * ~final_mask).unsqueeze(1)  # 动态区域权重降为10%

                target = target.view(-1, self.ht // self.down_scale, self.wd // self.down_scale, 2).permute(0, 3, 1,
                                                                                                            2).contiguous()
                weight = weight.view(-1, self.ht // self.down_scale, self.wd // self.down_scale, 2).permute(0, 3, 1,
                                                                                                            2).contiguous()
                # dynamic_ratio = final_mask.float().mean().item()
                # print(f"动态区域占比: {dynamic_ratio * 100:.2f}%")

                # 检查权重抑制是否应用
               # print("原始权重均值:", weight.mean().item())
                weight =  weight *  (~final_mask).unsqueeze(1)
               # print("抑制后权重均值:", weight.mean().item())
                #target = target * (~final_mask).unsqueeze(1)
                #static_weight = weight * (~final_mask).unsqueeze(1)
                #print("静态目标占比:", (target.abs() > 1e-3).float().mean().item())
                #target = target * (0.4 * final_mask + 0.6 * ~final_mask).unsqueeze(1) # 只考虑动态区域的目标优化
                #print("动态区域权重均值:", (0.1 * final_mask + 0.9 * ~final_mask).mean())
                # droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                #                   target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, False)
                # 使用非线性优化来处理动态区域的尺度与深度
                # static_weight = weight * (0.1 * final_mask + 0.9 * ~final_mask).unsqueeze(1)
                # droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                #                   target * (~final_mask).unsqueeze(1), static_weight, eta, ii, jj,
                #                   t0, t1, itrs, lm * 0.5, ep * 2, motion_only, False)
                # droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                #                   target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, False)
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                                  target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, False)
                # if use_mono:
                #     poses = lietorch.SE3(self.poses[:t1][None])
                #     disps = self.disps[:t1][None]
                #     dscales = self.dscales[:t1]
                #     disps, dscales, _ = JDSA(target, weight, eta, poses, disps, self.intrinsics[None], self.mono_disps,
                #                              dscales, ii, jj, self.mono_depth_alpha)
                #     self.disps[:t1] = disps[0]
                #     self.dscales[:t1] = dscales
                self.disps.clamp_(min=1e-5)  # 视差数值截断（防止负数）
                return True  # 返回优化成功标志

            ###################### 阶段2：深度尺度优化 ######################
            elif opt_type == "depth_scale":
                print("dspo2")
                # 初始化优化变量（SE3姿态、视差、尺度、位移）
                seg_masks = self.mask[ii]  # 假设已预生成并存储
                poses = lietorch.SE3(self.poses[None])  # 转换为李群表示
                disps = self.disps[None]
                scales = self.depth_scale
                shifts = self.depth_shift
                ignore_frames = 0  # 需要忽略的起始帧数
                # 添加缺失的维度（维度0和维度4）
                seg_masks = seg_masks.unsqueeze(0).unsqueeze(-1)  # 形状变为 [1, 96, 48, 64, 1]

                # 扩展维度4以匹配weight的尺寸2
                seg_masks = seg_masks.expand(-1, -1, -1, -1, 2)  # 最终形状 [1, 96, 48, 64, 2]
                print("seg_masks shape:", seg_masks.shape)  # 预期输出包含维度4为64
                print("weight shape:", weight.shape)  # 当前维度4为64
                weight=weight*seg_masks
                # 更新有效深度掩码
                self.update_valid_depth_mask(up=False)

                # 获取当前有效帧范围
                curr_idx = self.counter.value - 1
                mono_d = self.mono_disps[:curr_idx + 1]  # 单目深度估计
                est_d = self.disps[:curr_idx + 1]  # 当前视差估计
                valid_d = self.valid_depth_mask_small[:curr_idx + 1]  # 有效深度掩码

                # 计算尺度-位移对齐参数
                scale_t, shift_t, error_t = align_scale_and_shift(mono_d, est_d, valid_d)
                avg_disps = est_d.mean(dim=[1, 2])  # 平均视差计算

                # 更新全局尺度位移参数
                scales[:curr_idx + 1] = scale_t
                shifts[:curr_idx + 1] = shift_t

                # 数据缓存（优化过程使用）
                target_t, weight_t, eta_t, ii_t, jj_t = target, weight, eta, ii, jj

                ########### 单目深度异常值过滤（动态物体影响处理）###########
                if self.mono_thres:
                    # 构建异常掩码（包含：误差过大、NaN值、负尺度、有效区域不足等情况）
                    invalid_mono_mask = (error_t / avg_disps > self.mono_thres) | \
                                        (error_t.isnan()) | \
                                        (scale_t < 0) | \
                                        (valid_d.sum(dim=[1, 2]) < valid_d.shape[1] * valid_d.shape[2] * 0.5)

                    # 获取异常帧索引
                    invalid_mono_index, = torch.where(invalid_mono_mask.clone())

                    # 构建相关帧掩码（排除异常帧关联数据）
                    invalid_ii_mask = (ii < 0)  # 初始化为全False
                    idx_in_ii = torch.unique(ii)
                    for idx in invalid_mono_index:
                        invalid_ii_mask = invalid_ii_mask | (ii == idx) | (jj == idx)

                    # 应用掩码过滤异常数据
                    target_t = target[:, ~invalid_ii_mask]
                    weight_t = weight[:, ~invalid_ii_mask]
                    ii_t = ii[~invalid_ii_mask]
                    jj_t = jj[~invalid_ii_mask]

                    # 更新eta参数的有效性掩码
                    valid_eta_mask = torch.tensor([idx in torch.unique(ii_t) for idx in idx_in_ii]).to(self.device)
                    eta_t = eta[valid_eta_mask]

                ########### 执行迭代优化 ###########
                success = False
                for _ in range(itrs):  # 多轮次优化
                    if self.counter.value > ignore_frames and ii_t.shape[0] > 0:
                        # 调用带尺度位移的BA优化核心
                        poses, disps, wqs = thirdparty.glorie_slam.geom.ba.BA_with_scale_shift(
                            target_t, weight_t, eta_t, poses, disps,
                            self.intrinsics[None], ii_t, jj_t,
                            self.mono_disps[None],
                            scales[None], shifts[None],
                            self.valid_depth_mask_small[None], ignore_frames,
                            lm, ep, alpha=0.01
                        )
                        # 更新尺度位移参数
                        scales = wqs[0, :, 0]
                        shifts = wqs[0, :, 1]
                        success = True

                # 回写优化结果到类成员变量
                self.depth_scale = scales
                self.depth_shift = shifts
                self.disps = disps.squeeze(0)
                self.poses = poses.vec().squeeze(0)

                self.disps.clamp_(min=1e-5)  # 数值稳定性处理
                return success  # 返回优化状态

            else:
                raise NotImplementedError  # 未知优化类型异常
    def cuda_pgba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, verbose=False):
        from geom.ba import pose_retr

        poses = lietorch.Sim3(self.poses_sim3[:t1][None])

        # rel pose constraints
        rel_N = self.pgobuf.rel_N.value
        iip, jjp = self.pgobuf.rel_ii[:rel_N].cuda(), self.pgobuf.rel_jj[:rel_N].cuda()
        rel_poses = self.pgobuf.rel_poses[:rel_N].cuda()[None]
        infos = 1 / self.pgobuf.rel_covs[:rel_N].cuda()
        infos = torch.cat((infos, infos.min(dim=1, keepdim=True)[0]), dim=1)
        infos = infos.unsqueeze(2).expand(*infos.size(), infos.shape[-1]) * torch.eye(infos.shape[-1], device='cuda')[None]
        infos[torch.isnan(infos) | torch.isinf(infos)] = 0.
        seg_masks = self.mask[ii]  # 假设已预生成并存储
        seg_masks = ~seg_masks  # True表示动态区域
        weight = weight * (~seg_masks).unsqueeze(1)
        for _ in range(itrs):
            Hsp, vsp, pchi2, pchi2_scaled = global_relative_posesim3_constraints(iip, jjp, poses, rel_poses, infos, pw=1e-3)

            disps = self.disps[:t1][None]

            if verbose:
                coords, valid = pops.projective_transform(poses, disps, self.intrinsics[None], ii, jj)
                r = (target - coords).view(1, ii.shape[0], -1, 1)
                rw = .001 * (valid * weight).view(1, ii.shape[0], -1, 1)
                rchi2 = torch.sum((rw * r).transpose(2,3) @ r)
                print("- Chi2 error reproj: {:.5f} relpose: {:.5f} {:.5f}".format(rchi2.item(), pchi2.item(), pchi2_scaled.item()))

            B, P, ht, wd = disps.shape
            N = ii.shape[0]
            D = poses.manifold_dim

            ### 1: commpute jacobians and residuals ###
            coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
                poses, disps, self.intrinsics[None], ii, jj, jacobian=True)

            r = (target - coords).view(B, N, -1, 1)
            w = .001 * (valid * weight).view(B, N, -1, 1)

            ### 2: construct linear system ###
            Ji = Ji.reshape(B, N, -1, D)
            Jj = Jj.reshape(B, N, -1, D)
            wJiT = (w * Ji).transpose(2,3)
            wJjT = (w * Jj).transpose(2,3)

            Jz = Jz.reshape(B, N, ht*wd, -1)

            Hii = torch.matmul(wJiT, Ji)
            Hij = torch.matmul(wJiT, Jj)
            Hji = torch.matmul(wJjT, Ji)
            Hjj = torch.matmul(wJjT, Jj)
            Hs = torch.cat((Hii, Hij, Hji, Hjj))

            vi = torch.matmul(wJiT, r).squeeze(-1)
            vj = torch.matmul(wJjT, r).squeeze(-1)
            vs = torch.cat((vi, vj))

            Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
            Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

            w = w.view(B, N, ht*wd, -1)
            r = r.view(B, N, ht*wd, -1)
            wk = torch.sum(w*r*Jz, dim=-1)
            Ck = torch.sum(w*Jz*Jz, dim=-1)

            dx, dz = droid_backends.pgba(poses.data[0], self.disps, eta,
                                Hs, vs, Ei[0], Ej[0], Ck[0], wk[0],
                                Hsp, vsp, ii, jj, iip, jjp, t0, t1, lm, ep)
            # print(dx.mean(), dz.mean())
            poses = pose_retr(poses, dx[None], torch.arange(t0, t1))

        self.poses_sim3[:t1] = poses.data
        self.disps.clamp_(min=0.001, max=10)
    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False, opt_type="pose_depth",use_mono=False):
        if self.BA_type == "DSPO":
            success = self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,opt_type,use_mono)
            if not success:
                self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,opt_type,use_mono)
        elif self.BA_type == "DBA":
            self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,"pose_depth",use_mono)
        else:
            raise NotImplementedError


    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        print("mono",mono_depth.shape)
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        shift=max(shift,0)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            est_disp = self.disps_up[index].clone().to(device)  # [h, w]
            est_depth = 1.0 / (est_disp)
            depth_mask = self.valid_depth_mask[index].clone().to(device)
            c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        depths = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        np.savez(path,poses=poses,depths=depths,timestamps=timestamps,valid_depth_masks=valid_depth_masks)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)


    def eval_depth_l1(self, npz_path, stream, global_scale=None):
        # Compute Depth L1 error
        depth_l1_list = []
        depth_l1_list_max_4m = []
        mask_list = []

        # load from disk
        offline_video = dict(np.load(npz_path))
        video_timestamps = offline_video['timestamps']

        for i in range(video_timestamps.shape[0]):
            timestamp = int(video_timestamps[i])
            mask = self.valid_depth_mask[i]
            if mask.sum() == 0:
                print("WARNING: mask is empty!")
            mask_list.append((mask.sum()/(mask.shape[0]*mask.shape[1])).cpu().numpy())
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            # compute scale and shift for depth
            # load gt depth from stream
            depth_gt = stream[timestamp][2].to(self.device)
            mask = torch.logical_and(depth_gt > 0, mask)
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list.append(depth_l1.cpu().numpy())

            # update process but masking depth_gt > 4
            # compute scale and shift for depth
            mask = torch.logical_and(depth_gt < 4, mask)
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list_max_4m.append(depth_l1.cpu().numpy())

        return np.asarray(depth_l1_list).mean(), np.asarray(depth_l1_list_max_4m).mean(), np.asarray(mask_list).mean()

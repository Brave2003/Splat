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

from thirdparty.glorie_slam.motion_filter import MotionFilter
from thirdparty.glorie_slam.frontend import Frontend
from thirdparty.glorie_slam.backend import Backend
import torch
from colorama import Fore, Style
from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset
from src.utils.Printer import Printer,FontColor
from ultralytics import YOLO
from src.utils.datasets import get_dataset
import os
import numpy as np
import cv2
from utils.dynamic_mask_detector import run_frontend_motion_mask_and_save
class Tracker:
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.only_tracking = slam.only_tracking
        self.output = slam.save_dir
        # filter incoming frames so that there is enough motion
        self.frontend_window = self.cfg['tracking']['frontend']['window']
        filter_thresh = self.cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, self.cfg, thresh=filter_thresh, device=self.device)
        self.enable_online_ba = self.cfg['tracking']['frontend']['enable_online_ba']
        self.every_kf = self.cfg['mapping']['every_keyframe']
        # frontend process
        self.frontend = Frontend(self.net, self.video, self.cfg)
        #self.frontend = slam.frontenD
        self.online_ba = Backend(self.net,self.video, self.cfg)
        self.ba_freq = self.cfg['tracking']['backend']['ba_freq']
        self.printer:Printer = slam.printer

        #self.LC_data_queue=slam.LC_data_queue
    # def call_gs(self, viz_idx, dposes=None, dscale=None):
    #     data = {"is_keyframe": True,"video_idx":  viz_idx.to(device='cpu'),
    #             'timestamp':   self.video.timestamp[viz_idx].to(device='cpu'),
    #             'poses':    self.video.poses[viz_idx].to(device='cpu'),
    #             'images':   self.video.images[viz_idx.cpu()],
    #             #'normals':  self.video.normals[viz_idx.cpu()],
    #             'mask':     self.video.mask[viz_idx.cpu()],
    #             'depths':   1./self.video.disps_up[viz_idx.cpu()],
    #             'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu') * 8,
    #             'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
    #             'scale_updates': dscale.to(device='cpu') if dscale is not None else None,
    #              "end": False}
    #     return  data
    def run(self, stream: BaseDataset):
        '''
        Trigger the tracking process.
        1. 通过motion_filter检查当前帧与上一个关键帧之间是否有足够运动
        2. 使用frontend进行局部集束调整，估计相机姿态和深度图，
            如果在局部BA后发现当前关键帧与前一关键帧距离过近则删除当前关键帧
        3. 通过backend定期执行在线全局BA
        4. 将估计的姿态和深度发送给mapper，
            并等待mapper完成当前的建图优化
        '''
        prev_kf_idx = 0  # 上一个关键帧索引
        curr_kf_idx = 0  # 当前关键帧索引
        prev_ba_idx = 0  # 上次执行BA的关键帧索引
        last = 0
        number_of_kf = 0  # 关键帧计数器
        intrinsic = stream.get_intrinsic()  # 获取相机的内参
        # for (timestamp, image, _, _) in tqdm(stream):
        dposes, dscale= None,None
        self.kf_timestamps = [0]
        self.kf_video_idxs = [curr_kf_idx]
        for i in range(len(stream)):
            timestamp, image, depth, pose, mask ,normal,mono,static_msk= stream[i]  # 从数据流中获取时间戳、图像、掩码等数据
            #print("pose",pose)
            #print("mask",mask)
            with torch.no_grad():  # 禁用梯度计算
                ### 检查是否有足够的运动
                self.motion_filter.track(timestamp, image,depth,pose, intrinsic, mask)  # 运动滤波器跟踪
                # 局部集束调整
                self.frontend()  # 执行前端处理

            curr_kf_idx = self.video.counter.value-1   # 获取当前关键帧索引
            # self.pgba=self.cfg["mapping"]["Training"]["pgba"]["active"]
            # if len(viz_idx) and self.pgba:
            #     dposes, dscale = self.video.pgobuf.run_pgba(self.LC_data_queue)
                #if dposes is not None:
                    #data=self.call_gs(torch.arange(0, curr_kf_idx, device='cuda'), dposes[:-1], dscale[:-1])

            # if len(viz_idx):
            #     data=self.call_gs(viz_idx)
            #print("mask",self.video.mask_ori[curr_kf_idx])
            # output_dir = "try mask_ori_output"
            # os.makedirs(output_dir, exist_ok=True)
            # mask_np = self.video.mask_ori[curr_kf_idx].cpu().numpy().astype(np.uint8) * 255  # True->255, False->0
            #
            # # 保存图像（文件名按索引命名）
            # output_path = os.path.join(output_dir, f"mask_ori_{curr_kf_idx:04d}.png")
            # cv2.imwrite(output_path, mask_np)
            #del mask_np
            if curr_kf_idx != prev_kf_idx:
                number_of_kf += 1  # 关键帧计数增加
                self.kf_timestamps.append(i)
                self.kf_video_idxs.append(curr_kf_idx)
                enable_mask = self.cfg.get("enable_dynamic_mask_detection", False)
                if enable_mask and len(self.kf_timestamps) >= 2 and self.only_tracking:
                    min_frames_in_interval = 10
                    last_ts = self.kf_timestamps[-1]
                    start_i = 0
                    for hist_i in range(len(self.kf_timestamps) - 1, -1, -1):
                        if self.kf_timestamps[hist_i] <= last_ts - (min_frames_in_interval - 1):
                            start_i = hist_i
                            break
                    ts_window = self.kf_timestamps[start_i:]
                    vid_window = self.kf_video_idxs[start_i:]
                    span = (ts_window[-1] - ts_window[0] + 1) if len(ts_window) else 0
                    if span >= min_frames_in_interval:
                        try:
                            run_frontend_motion_mask_and_save(
                                stream=stream,
                                keyframe_timestamps=ts_window,
                                keyframe_video_idxs=vid_window,
                                save_dir=self.output,
                                device=torch.device(self.device),
                                config={
                                    "enable_dynamic_mask_detection": True,
                                    "seg_chair": getattr(stream, "seg_chair", False),
                                },
                            )
                        except Exception as exc:
                            self.printer.print(
                                f"Motion mask update failed at frame {timestamp}: {exc}",
                                FontColor.TRACKER,
                            )
                if self.frontend.is_initialized:
                    # 每间隔ba_freq个关键帧执行在线全局BA
                    if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq and  self.only_tracking :
                        self.printer.print(f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}",
                                           FontColor.TRACKER)
                        self.online_ba.dense_ba(2)  # 执行密集全局BA
                        prev_ba_idx = curr_kf_idx  # 更新上次BA的关键帧索引
                    # 非纯跟踪模式且达到关键帧间隔
                    if (not self.only_tracking) and (number_of_kf % self.every_kf == 0) and (i-last >=3) :#and i != 180 and i!=204: #and (curr_kf_idx>=self.dystart) :#and timestamp>70:
                        # 通知mapper当前姿态和深度估计已完成
                        last=i
                        self.pipe.send({"is_keyframe": True,"video_idx":  curr_kf_idx,
                                        'timestamp':timestamp,"end": False})

                        self.pipe.recv()  # 等待mapper确认
            prev_kf_idx = curr_kf_idx  # 更新上一个关键帧索引
            self.printer.update_pbar()  # 更新进度条

        # 非纯跟踪模式时发送结束信号
        if not self.only_tracking:
            self.pipe.send({"is_keyframe": True, "video_idx": None,
                            "timestamp": None, "end": True})

                

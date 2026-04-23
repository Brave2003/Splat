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

import os
import torch
import numpy as np
from collections import OrderedDict
import torch.multiprocessing as mp
from evo.core.trajectory import PosePath3D
from evo.core import lie_algebra as lie
from thirdparty.glorie_slam.modules.droid_net import DroidNet
from thirdparty.glorie_slam.depth_video import DepthVideo
from thirdparty.glorie_slam.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed,update_cam
from src.utils.Printer import Printer,FontColor
from src.utils.eval_traj import kf_traj_eval,full_traj_eval
from src.utils.eval_utils import eval_rendering
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from thirdparty.glorie_slam.backend import Backend
from torch.cuda import memory_reserved, empty_cache
#from thirdparty.glorie_slam.pgo_buffer import PGOBuffer
# from thirdparty.glorie_slam.frontend import Frontend
# from torch.multiprocessing import Process
# from src.visualization import droid_visualization
#from utils.eval_utils import save_gaussians
import time

def get_available_gpus():
    """返回可用GPU列表，按显存剩余量排序"""
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        mem_reserved = memory_reserved(i)
        mem_total = torch.cuda.get_device_properties(i).total_memory
        mem_free = mem_total - mem_reserved
        available_gpus.append((i, mem_free))

    # 按显存剩余量从大到小排序
    available_gpus.sort(key=lambda x: x[1], reverse=True)
    return [gpu[0] for gpu in available_gpus]


# class SharedVisualizationData:
#     def __init__(self, video):
#         self.video = video
#         self.lock = mp.Lock()
#         self._setup_shared_arrays()
#         self.initialized = mp.Value('b', False)
#
#         # 精确记录原始数据形状
#         self.original_shapes = {
#             'poses': (video.poses.shape[0], 7),
#             'disps': (video.disps.shape[0], video.ht // video.down_scale, video.wd // video.down_scale),
#             'disps_up': (video.disps_up.shape[0], video.ht, video.wd),
#             'images': (video.images.shape[0], 3, video.ht, video.wd),
#             'intrinsics': (video.intrinsics.shape[0], 4)
#         }
#
#     def _setup_shared_arrays(self):
#         # 完全按照video对象的实际形状初始化
#         buffer_size = self.video.poses.shape[0]
#         ht = self.video.ht
#         wd = self.video.wd
#         ht_small = ht // self.video.down_scale
#         wd_small = wd // self.video.down_scale
#
#         # 主共享数组
#         self.shared_poses = mp.RawArray('d', buffer_size * 7)
#         self.shared_disps = mp.RawArray('f', buffer_size * ht_small * wd_small)
#         self.shared_disps_up = mp.RawArray('f', buffer_size * ht * wd)
#         self.shared_images = mp.RawArray('B', buffer_size * 3 * ht * wd)
#         self.shared_intrinsics = mp.RawArray('f', buffer_size * 4)
#
#         # 辅助数组
#         self.shared_timestamps = mp.RawArray('d', buffer_size)
#         self.shared_counter = mp.RawValue('i', 0)
#
#     def get_data(self):
#         with self.lock:
#             if not hasattr(self.video, 'disps_up'):
#                 print("错误：video对象缺少disps_up属性！")
#                 return None
#
#                 # 确保深度图数据有效
#             if torch.max(self.video.disps_up) < 1e-5:
#                 print("警告：深度图数据全零！")
#                 return None
#
#                 # 强制转换数据类型（修复常见问题）
#             images = self.video.images[:self.video.counter.value].cpu().numpy()
#             if images.dtype != np.uint8:
#                 images = (images * 255).astype(np.uint8)
#             print(f"Current counter: {self.video.counter.value}")
#             print(f"Poses shape: {self.video.poses.shape}")
#             print(f"Disps shape: {self.video.disps_up.shape}")
#             print(f"Images dtype: {self.video.images.dtype}")
#             current_count = min(self.video.counter.value, self.video.poses.shape[0])
#             if current_count <= 0:
#                 return None
#
#             try:
#                 # 获取数据并保持原始形状
#                 poses = self.video.poses[:current_count].cpu().numpy()
#                 disps = self.video.disps[:current_count].cpu().numpy()
#                 disps_up = self.video.disps_up[:current_count].cpu().numpy()
#                 images = self.video.images[:current_count].cpu().numpy()
#                 intrinsics = self.video.intrinsics[:current_count].cpu().numpy()
#                 timestamps = self.video.timestamp[:current_count].cpu().numpy()
#
#                 # 更新共享内存
#                 self._update_array(self.shared_poses, poses.reshape(-1))
#                 self._update_array(self.shared_disps, disps.reshape(-1))
#                 self._update_array(self.shared_disps_up, disps_up.reshape(-1))
#                 self._update_array(self.shared_images, images.reshape(-1))
#                 self._update_array(self.shared_intrinsics, intrinsics.reshape(-1))
#                 self._update_array(self.shared_timestamps, timestamps.reshape(-1))
#
#                 with self.shared_counter.get_lock():
#                     self.shared_counter.value = current_count
#
#                 self.initialized.value = True
#
#                 return {
#                     'poses': poses,
#                     'disps': disps,
#                     'disps_up': disps_up,
#                     'images': images,
#                     'intrinsics': intrinsics,
#                     'timestamps': timestamps,
#                     'counter': current_count,
#                     'ht': self.video.ht,
#                     'wd': self.video.wd,
#                     'down_scale': self.video.down_scale
#                 }
#
#             except Exception as e:
#                 print(f"Data sharing error: {str(e)}")
#                 return None
#
#     def _update_array(self, dest, src):
#         """安全更新共享数组"""
#         dest_np = np.frombuffer(dest, dtype=src.dtype)
#         np.copyto(dest_np[:src.size], src.ravel())
class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        # 调用父类的构造函数
        super(SLAM, self).__init__()
        # 从配置字典中获取设备类型，保存到self.device
        self.cfg = cfg
        self.disable_vis = cfg["disable_vis"]
        self.device = cfg['device']
        # 获取是否打印详细信息的标志
        self.verbose: bool = cfg['verbose']
        # 获取是否仅执行跟踪的标志
        self.only_tracking: bool = cfg['only_tracking']
        # 初始化日志记录器为None
        self.logger = None
        # 设置保存输出结果的目录路径
        self.save_dir = cfg["data"]["output"] + '/' + cfg['scene']
        # 如果路径不存在，则创建该目录
        os.makedirs(self.save_dir, exist_ok=True)
        # 从配置中更新相机的相关参数（如宽度、高度、焦距等）
        self.H, self.W, \
            self.fx, self.fy, \
            self.cx, self.cy = update_cam(cfg)
        # 初始化DroidNet模型
        self.droid_net: DroidNet = DroidNet()
        # 初始化打印器，用于在后台处理打印信息
        self.printer = Printer(len(stream))  # 使用额外的进程来打印所有信息
        # 加载预训练模型
        self.load_pretrained(cfg)
        # 将DroidNet模型移动到指定设备（如GPU），并设置为评估模式
        self.droid_net.to(self.device).eval()
        # 共享内存，确保多个进程可以访问模型
        self.droid_net.share_memory()
        # 创建一个整型Tensor，用于记录当前运行的线程数量
        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        # 创建一个整型Tensor，用于记录所有线程是否都已触发的标志
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()
        # 初始化深度视频对象，包含了视频处理和打印信息功能
        self.video = DepthVideo(cfg, self.printer)
        # 初始化后端处理对象（主要用于视频处理的后端计算）
        self.ba = Backend(self.droid_net, self.video, self.cfg)
        # 初始化姿态轨迹填充器，用于补全非关键帧的位姿
        self.traj_filler = PoseTrajectoryFiller(net=self.droid_net, video=self.video,
                                            printer=self.printer, device=self.device)
        #self.frontenD = Frontend(self.droid_net, self.video, self.cfg)
        # 初始化跟踪器和映射器为None，稍后会进行实例化
        available_gpus = get_available_gpus()
        if len(available_gpus) >= 2:
            self.mapping_device = torch.device(f'cuda:{available_gpus[1]}')
        else:
            self.mapping_device = torch.device(f'cuda:{available_gpus[0]}')  # 如果没有其他 GPU，就用主 GPU
        self.tracker: Tracker = None
        self.mapper: Mapper = None

        # 将数据流（BaseDataset）保存到实例变量中，供后续使用
        self.stream = stream

    def load_pretrained(self, cfg):
        droid_pretrained = cfg['tracking']['pretrained']
        state_dict = OrderedDict([
            (k.replace('module.', ''), v) for (k, v) in torch.load(droid_pretrained).items()
        ])
        state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
        state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
        state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
        state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()

    def tracking(self, pipe):
        self.tracker = Tracker(self, pipe)
        self.printer.print('Tracking Triggered!',FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f'{self.save_dir}/mono_priors/depths', exist_ok=True)

        while(self.all_trigered < self.num_running_thread):
            pass
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print('Tracking Done!',FontColor.TRACKER)
        if self.only_tracking:
            self.terminate()
    
    def mapping(self, pipe):
        if self.only_tracking:
            self.all_trigered += 1
            return
        #print("pipe",pipe)
            # 初始化 Mapper 并移动到 mapping_device
        # if hasattr(pipe, 'to'):
        #     pipe = pipe.to(self.mapping_device)
        # elif isinstance(pipe, dict):
        #     pipe = {k: v.to(self.mapping_device) for k, v in pipe.items()}
        self.mapper = Mapper(self, pipe)
        self.printer.print('Mapping Triggered!', FontColor.MAPPER)

        # 等待所有线程就绪
        self.all_trigered += 1
        while self.all_trigered < self.num_running_thread:
            pass

        # 运行 Mapper（确保输入数据也在正确的设备上）
        self.mapper.run(self.stream)
        self.printer.print('Mapping Done!', FontColor.MAPPER)

        self.terminate()
        

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)
        self.ba = Backend(self.droid_net,self.video,self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!",FontColor.TRACKER)


    def terminate(self):
        """ fill poses for non-keyframe images and evaluate """
        if self.cfg['tracking']['backend']['final_ba'] and self.cfg['mapping']['eval_before_final_ba']:
            self.video.save_video(f"{self.save_dir}/video.npz")
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",self.stream,self.logger,self.printer)
            except Exception as e:
                self.printer.print(e,FontColor.ERROR)

            if not self.only_tracking: 
                # prepare aligned camera list of mapped frames
                traj_est_aligned = []
                cams = self.mapper.cameras
                for kf_idx in self.mapper.video_idxs:
                    traj_est_aligned.append(np.linalg.inv(gen_pose_matrix(cams[kf_idx].R, cams[kf_idx].T)))

                traj_est_aligned = PosePath3D(poses_se3=traj_est_aligned)
                traj_est_aligned.scale(global_scale)
                traj_est_aligned.transform(lie.se3(r_a, t_a))
                rendering_result = eval_rendering(
                    self.mapper,
                    self.save_dir,
                    iteration="before_refine",
                    monocular=True,
                    mesh=self.cfg["meshing"]["mesh_before_final_ba"],
                    traj_est_aligned=list(traj_est_aligned.poses_se3),
                    global_scale=global_scale,
                    scene=self.cfg['scene'],
                    eval_mesh=True if self.cfg['dataset'] == 'replica' else False,
                    gt_mesh_path=self.cfg['meshing']['gt_mesh_path']
                )
        if self.cfg['tracking']['backend']['final_ba']:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        try:
            ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                f"{self.save_dir}/video.npz",
                f"{self.save_dir}/traj",
                "kf_traj",self.stream,self.logger,self.printer)
        except Exception as e:
            self.printer.print(e,FontColor.ERROR)

        if not self.only_tracking:
            if self.cfg['tracking']['backend']['final_ba']:
                # The final refine method includes the final update of the poses and depths
                self.mapper.final_refine(iters=self.cfg["mapping"]["final_refine_iters"]) # this performs a set of optimizations with RGBD loss to correct

            # prepare aligned camera list of mapped frames
            traj_est_aligned = []
            cams = self.mapper.cameras
            for kf_idx in self.mapper.video_idxs:
                traj_est_aligned.append(np.linalg.inv(gen_pose_matrix(cams[kf_idx].R, cams[kf_idx].T)))

            traj_est_aligned = PosePath3D(poses_se3=traj_est_aligned)
            traj_est_aligned.scale(global_scale)
            traj_est_aligned.transform(lie.se3(r_a, t_a))
            # evaluate the metrics
            rendering_result = eval_rendering(
                self.mapper,
                self.save_dir,
                iteration="after_refine",
                monocular=True,
                mesh=self.cfg["meshing"]["mesh"],
                traj_est_aligned=list(traj_est_aligned.poses_se3),
                global_scale=global_scale,
                scene=self.cfg['scene'],
                eval_mesh=True if self.cfg['dataset'] == 'replica' else False,
                gt_mesh_path=self.cfg['meshing']['gt_mesh_path']
            )

        # evaluate depth error
        self.printer.print("Evaluate sensor depth error with per frame alignment",FontColor.EVAL)
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(f"{self.save_dir}/video.npz", self.stream)
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m),FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage),FontColor.EVAL)

        self.printer.print("Evaluate sensor depth error with global alignment",FontColor.EVAL)
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(f"{self.save_dir}/video.npz", self.stream, global_scale)
        self.printer.print("Depth L1: " + str(depth_l1_g),FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m_g),FontColor.EVAL)

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f'{self.save_dir}/depth_stats.txt'
        integers = {
            'depth_l1': depth_l1,
            'depth_l1_global_scale': depth_l1_g,
            'depth_l1_mask_4m': depth_l1_max_4m,
            'depth_l1_mask_4m_global_scale': depth_l1_max_4m_g,
            'Average frame coverage': coverage, # How much of each frame uses depth from droid (the rest from Omnidata)
            'traj scaling': global_scale,
            'traj rotation': r_a,
            'traj translation': t_a,
            'traj stats': ate_statistics
        }
        # Write to the file
        with open(file_path, 'w') as file:
            for label, number in integers.items():
                file.write(f'{label}: {number}\n')

        self.printer.print(f'File saved as {file_path}',FontColor.EVAL)

        full_traj_eval(self.traj_filler,
                       f"{self.save_dir}/traj",
                       "full_traj",
                       self.stream, self.logger, self.printer)

        self.printer.print("Metrics Evaluation Done!",FontColor.EVAL)

    def run(self):
        m_pipe, t_pipe = mp.Pipe()

        # 初始化进程列表：跟踪(tracking)与建图(mapping)
        processes = [
            mp.Process(target=self.tracking, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe,)),
        ]

        # 如果启用位姿图优化（PGBA），添加相关进程
        # self.pgba =self.cfg["mapping"]["Training"]["pgba"]["active"]
        # if self.pgba:
        #     # 初始化PGBA缓冲区和闭环检测队列
        #     self.video.pgobuf = PGOBuffer(
        #         self.droid_net, self.video, self.frontenD,
        #         self.cfg["mapping"]["Training"]["pgba"]  # 从配置读取参数
        #     )
        #     self.LC_data_queue = mp.Queue()  # 使用多进程安全队列
        #     self.video.pgobuf.set_LC_data_queue(self.LC_data_queue)
        #
        #     # 创建PGBA后台进程并加入进程列表
        #     self.mp_backend = mp.Process(target=self.video.pgobuf.spin)
        #     processes.append(self.mp_backend)

        # 启动所有进程
        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 终止打印线程（假设独立于上述进程）
        self.printer.terminate()
        # core_processes = [
        #     mp.Process(target=self.tracking, args=(t_pipe,)),
        #     mp.Process(target=self.mapping, args=(m_pipe,)),
        # ]
        # for p in core_processes:
        #     p.start()
        #
        # # 延迟启动可视化进程（确保数据流已建立）
        # if not self.disable_vis:
        #     vis_process = mp.Process(target=droid_visualization, args=(self.video,))
        #     time.sleep(2)  # 等待SLAM核心初始化
        #     vis_process.start()
        #     processes = core_processes + [vis_process]

        # # 等待核心进程初始化
        # time.sleep(3)  # 更长的等待时间确保SLAM初始化完成
        #
        # # 可视化进程
        # vis_process = None
        # if not self.disable_vis:
        #     try:
        #         # 确保video对象已初始化
        #         while self.video.counter.value <= 0:
        #             time.sleep(0.5)
        #
        #         shared_data = SharedVisualizationData(self.video)
        #         vis_process = mp.Process(
        #             target=slam_realtime_visualization,
        #             args=(shared_data, str(self.device)),
        #             daemon=True
        #         )
        #         vis_process.start()
        #     except Exception as e:
        #         print(f"Failed to start visualizer: {str(e)}")
        #         self.disable_vis = True
        #
        # try:
        #     # 监控进程状态
        #     while True:
        #         # 检查核心进程
        #         if not all(p.is_alive() for p in processes):
        #             dead = [p for p in processes if not p.is_alive()]
        #             print(f"Critical process died: {[p.name for p in dead]}")
        #             break
        #
        #         # 检查可视化进程
        #         if vis_process and not vis_process.is_alive():
        #             print("Visualizer terminated")
        #             vis_process = None
        #             if not self.disable_vis:
        #                 try:
        #                     shared_data = SharedVisualizationData(self.video)
        #                     vis_process = mp.Process(
        #                         target=slam_realtime_visualization,
        #                         args=(shared_data, str(self.device)),
        #                         daemon=True
        #                     )
        #                     vis_process.start()
        #                 except:
        #                     self.disable_vis = True
        #
        #         time.sleep(1)
        #
        # except KeyboardInterrupt:
        #     print("\nShutting down SLAM...")
        # finally:
        #     # 终止顺序很重要
        #     for p in processes:
        #         if p.is_alive():
        #             p.terminate()
        #     if vis_process and vis_process.is_alive():
        #         vis_process.terminate()
        #
        #     for p in processes:
        #         p.join(timeout=2.0)
        #     if vis_process:
        #         vis_process.join(timeout=1.0)


def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose

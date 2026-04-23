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

import torch
import lietorch
import cv2
import os
import numpy as np
import thirdparty.glorie_slam.geom.projective_ops as pops
from thirdparty.glorie_slam.modules.droid_net import CorrBlock
from src.mono_estimators import get_mono_depth_estimator,predict_mono_depth
from src.utils.datasets import load_mono_depth
from torchvision import transforms
from thirdparty.mono_priors.omnidata.modules.midas.omnidata import OmnidataModel
import torch.nn.functional as F
class MotionFilter:
    """ This class is used to filter incoming frames and extract features 
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        self.omni_dep=None
        self.dynamic_later=cfg["tracking"]["motion_filter"]["dynamic_later"]
        self.dynamic_thresh=cfg["tracking"]["motion_filter"]["dynamic_thresh"]
        self.dynamic_frame=cfg["tracking"]["motion_filter"]["dynamic_frame"]
        if cfg["mono_prior"]["predict_online"]:
            #self.mono_depth_estimator,self.mono_depth_estimator1 = get_mono_depth_estimator(cfg)
            self.mono_depth_estimator = get_mono_depth_estimator(cfg)
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)
    # def prior_extractor(self, im_tensor):
    #     input_size = im_tensor.shape[-2:]
    #     trans_totensor = transforms.Compose([transforms.Resize((512, 512), antialias=True)])
    #     im_tensor = trans_totensor(im_tensor).cuda()
    #     if self.omni_dep is None:
    #         #self.omni_dep = OmnidataModel('depth', 'pretrained/omnidata_dpt_depth_v2.ckpt', device="cuda:0")
    #         self.omni_normal = OmnidataModel('normal', 'pretrained/omnidata_dpt_normal_v2.ckpt', device="cuda:0")
    #     # depth = self.omni_dep(im_tensor)[None]*50
    #     # depth = F.interpolate(depth, input_size, mode='bicubic')
    #     # depth = depth.float().squeeze()
    #     normal = self.omni_normal(im_tensor) * 2.0 - 1.0
    #     normal = F.interpolate(normal, input_size, mode='bicubic')
    #     normal = normal.float().squeeze()
    #     return  normal
    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth,pose,intrinsics=None,mask=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # normalize images
        inputs = image[None, :, :].to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        # extract features
        gmap = self.__feature_encoder(inputs)
        # weight_prior = 0.4  # 给 prior_depth 的权重
        # weight_mono = 0.6  # 给 mono_depth 的权重
        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            # print("shape",image.shape)
            if self.cfg["mono_prior"]["predict_online"]:
                save_dir = os.path.join("output", "depth_out_try")
                #normal= self.prior_extractor(inputs[0])
                #print(f"Depth min/max: {prior_depth.min()}, {prior_depth.max()}")
                #os.makedirs(save_dir, exist_ok=True)
                #file_name = f"depth_{tstamp:04d}.png"
               # file_path = os.path.join(save_dir, file_name)
               # cv2.imwrite(file_path, prior_depth.cpu().numpy())
                #depth = depth.to(self.device)
                mono_depth=predict_mono_depth(self.mono_depth_estimator,tstamp,depth,image,self.cfg,self.device,mask)
                #depth = depth.to(mono_depth.device)
                #depth=mono_depth*mask+depth*(~mask)
                #mono_depth=0.2 * prior_depth + 0.8 * mono_depth
            else:
                mono_depth = load_mono_depth(tstamp, os.path.join(self.cfg['data']['output'], self.cfg['scene']))
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0],mask,pose)
        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)
            # depth = predict_mono_depth(self.mono_depth_estimator, tstamp, image, self.cfg, self.device)
            # save_dir = os.path.join("output", "depth")
            # os.makedirs(save_dir, exist_ok=True)
            # file_name = f"depth_{tstamp:04d}.png"
            # file_path = os.path.join(save_dir, file_name)
            # depth_image = depth.cpu().numpy()
            # min_depth = depth_image.min()
            # max_depth = depth_image.max()
            # mono= predict_mono_depth(self.mono_depth_estimator, tstamp, image, self.cfg, self.device)
            # save_dir = os.path.join("output", "monodepth")
            # os.makedirs(save_dir, exist_ok=True)
            # file_name = f"depth_{tstamp:04d}.png"
            # file_path = os.path.join(save_dir, file_name)
            # depth_image = mono.cpu().numpy()
            # # print("depth shape",depth_image.shape)
            # min_depth = depth_image.min()
            # max_depth = depth_image.max()
            # #
            # # # 归一化到 0-255 范围内（8位图像）
            # depth_normalized_8bit = ((depth_image - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
            # #
            # # # 或者，归一化到 0-65535 范围内（16位图像）
            # # # depth_normalized_16bit = ((depth_image - min_depth) / (max_depth - min_depth) * 65535).astype(np.uint16)
            # #
            # # # 将深度图保存为文件
            # cv2.imwrite(file_path, depth_normalized_8bit)  # 保存为 8 位图像
            # # # 归一化到 0-255 范围内（8位图像）
            # depth_normalized_8bit = ((depth_image - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
            # #
            # # # 或者，归一化到 0-65535 范围内（16位图像）
            # # # depth_normalized_16bit = ((depth_image - min_depth) / (max_depth - min_depth) * 65535).astype(np.uint16)
            # #
            # # # 将深度图保存为文件
            # cv2.imwrite(file_path, depth_normalized_8bit)  # 保存为 8 位图像
            # # # cv2.imwrite(file_path, depth_normalized_16bit)  # 保存为 16 位图像
            # print("depth.shape",depth.shape)
            # print(f"深度图已保存为: {file_path}")
            # # check motion magnitue / add new frame to video
            if self.dynamic_later and tstamp > self.dynamic_frame:
                self.thresh=self.dynamic_thresh
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                if self.cfg["mono_prior"]["predict_online"]:
                    #normal = self.prior_extractor(inputs[0])
                    depth = depth.to(self.device)
                    mono_depth = predict_mono_depth(self.mono_depth_estimator, tstamp,depth, image, self.cfg, self.device,mask)
                #     #depth = depth.to(mono_depth.device)
                #     #depth = mono_depth * mask + depth * (~mask)
                #     #mono_depth = 0.2 * prior_depth + 0.8* mono_depth
                else:
                    mono_depth = load_mono_depth(tstamp, os.path.join(self.cfg['data']['output'], self.cfg['scene']))
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0],mask,pose)

            else:
                self.count += 1

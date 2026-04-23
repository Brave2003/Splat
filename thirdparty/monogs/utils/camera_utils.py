# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md

import torch
from torch import nn
import numpy as np
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from thirdparty.monogs.utils.slam_utils import image_gradient, image_gradient_mask
import os
import cv2
from RAFT.raft import RAFT
from RAFT.utils import flow_viz
from RAFT.utils.utils import InputPadder

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained")

class raft_param():
    def __init__(self):
        self.dataset_path =  "/path/to/dataset"
        self.model = os.path.join(PRETRAINED_DIR, "raft-things.pth")
        #self.model = "gma-things.pth"  # ʹ��RAFTGMA���ƹ����ٶ���Щ
        self.small = False  # Сģ��
        self.mixed_precision = False  # ��ʹ�û�Ͼ���
def process_image(color):
    resized_image_rgb = color
    gt_image = resized_image_rgb
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gt_image, gray_image
class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        time,
        normal,
        motion_mask=None,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        # Absolute pose as W2C
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None
        self.normal=normal
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width
        self.time = time
        self.fid = torch.Tensor(np.array([time])).to(device)

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
        self.motion_mask = motion_mask
        self.attn_splits_list=2,
        self.corr_radius_list=-1,
        self.prop_radius_list=-1,
        self.pred_bidir_flow=True,
        self.rendered_mask = None
        self.save_mask = False
        self.flow = None
        self.flow_back = None
        self.mask_fwd = None
        self.mask_bwd = None
        self.epi_error = None
    @staticmethod
    def init_from_dataset(dataset, video_idx,idx,data, projection_matrix):
        _,_, depth_gtd, _, motion_mask,normal,mono,static_msk = dataset[idx]
        time = idx / (len(dataset) - 1)


        return Camera(
            data["idx"],
            data["gt_color"],
            #depth_gtd,
            data["glorie_depth"], # depth as in GlORIE-SLAM
            data["glorie_pose"], # pose as in GlORIE-SLAM
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.H_out,
            dataset.W_out,
            time,
            normal,
            motion_mask,
            device=dataset.device,
        )
    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)
    def compute_grad_mask(self, config):
        edge_threshold = config["mapping"]["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)


        median_img_grad_intensity = img_grad_intensity.median()

        self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
        )
    def get_image(self):
        gt_image, gray_image = process_image(self.original_image)
        return gt_image.cuda(), gray_image.cuda()
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width / scale), int(self.image_height / scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
            [(ix - self.cx / scale) / self.fx * scale,
             (iy - self.cy / scale) / self.fy * scale,
             torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.fx / scale, 0, self.cx / scale],
                        [0, self.fy / scale, self.cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.fx, 0, -self.cx/self.fx],
                            [0, scale/self.fy, -self.cy/self.fy],
                            [0, 0, 1]]).cuda()
        return K_T
    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None


    def reproject_mask(self, dataset, cam_0):
        # if self.uid==0:  # ��0֡��ȥ����̬���巴����Ч�������
        #    return None
        gt_depth_0 = torch.from_numpy(np.copy(cam_0.depth)).to(
            dtype=torch.float32, device=self.device
        )[None]
        W, H = dataset.width, dataset.height

        if torch.all(~((gt_depth_0[0] > 0) & (~cam_0.motion_mask))):
            return torch.ones((H, W), device=self.device, dtype=torch.bool)

        valid_depth_indices = torch.where((gt_depth_0[0] > 0) & (~cam_0.motion_mask))  # ��ȡgt_depth_0 ����0������
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        w2c = getWorld2View2(cam_0.R, cam_0.T)
        pts = self.get_pointcloud(gt_depth_0, dataset, w2c, valid_depth_indices)
        curr_w2c = getWorld2View2(self.R, self.T)
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (curr_w2c @ pts4.T).T[:, :3]

        intrinsics = torch.eye(3).to(device=self.device)
        intrinsics[0][2] = dataset.cx
        intrinsics[1][2] = dataset.cy
        intrinsics[0][0] = dataset.fx
        intrinsics[1][1] = dataset.fy
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2].long()
        valid_projected_pts = projected_pts[
            (projected_pts[:, 0] >= 0) & (projected_pts[:, 0] < W) & (projected_pts[:, 1] >= 0) & (projected_pts[:, 1] < H)]
        rendered_mask = torch.zeros((H, W), device=self.device, dtype=torch.bool)
        rendered_mask[valid_projected_pts[:, 1], valid_projected_pts[:, 0]] = True

        import torch.nn.functional as F
        kernel = torch.ones((3, 3), device=self.device, dtype=torch.bool)  # ʹ�� 3x3 �ľ����
        for _ in range(3):
            dilated_mask = F.conv2d(rendered_mask.unsqueeze(0).unsqueeze(0).float(),
                                    kernel.unsqueeze(0).unsqueeze(0).float(), padding=1)
            # �����ת���� bool ���ͣ���ȥ�������ά��
            rendered_mask = dilated_mask.squeeze().bool()

        if not self.save_mask and False:
            import matplotlib.pyplot as plt
            import os
            mask_image = (~rendered_mask.cpu().numpy().astype('uint8')) * 255
            plt.imshow(mask_image)
            plt.axis("off")
            os.makedirs("results/render_mask/", exist_ok=True)
            plt.savefig(f"results/render_mask/mask_{self.uid}.png", bbox_inches="tight", pad_inches=0)
            self.save_mask = True

        return ~rendered_mask


    def keyframe_selection_overlap(self, dataset, cam, time, pixels=1600, pose_window=3):
        intrinsics = torch.eye(3).to(device=self.device)
        intrinsics[0][2] = dataset.cx
        intrinsics[1][2] = dataset.cy
        intrinsics[0][0] = dataset.fx
        intrinsics[1][1] = dataset.fy
        W, H = dataset.W_out, dataset.H_out
        gt_depth_0 = torch.from_numpy(np.copy(self.depth)).to(
            dtype=torch.float32, device=self.device
        )[None]

        valid_depth_indices = torch.where((gt_depth_0[0] > 0))
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        w2c = getWorld2View2(self.R, self.T)
        pts = self.get_pointcloud(gt_depth_0, dataset, w2c, valid_depth_indices)
        list_keyframe = []
        for cam_idx, viewpoint in cam.items():
            if cam_idx >= time:
                continue
            est_w2c = getWorld2View2(viewpoint.R, viewpoint.T)
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]
            # Filter out the points that are outside the image
            edge = 20
            mask = (projected_pts[:, 0] < W - edge) * (projected_pts[:, 0] > edge) * \
                   (projected_pts[:, 1] < H - edge) * (projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum() / projected_pts.shape[0]
            list_keyframe.append(
                {'id': cam_idx, 'percent_inside': percent_inside})

            # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        win = 8 - pose_window
        if len(selected_keyframe_list) >= win:
            selected_keyframe_list = list(np.random.permutation(
                np.array(selected_keyframe_list))[:win])
        elif len(selected_keyframe_list) < win:
        # Need to add extra frames to make it 5
            remaining_needed = win - len(selected_keyframe_list)
        # Get all available frames (including those with percent_inside <= 0)
            all_frames = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        # Exclude already selected frames
            available_frames = [f for f in all_frames if f not in selected_keyframe_list]
        # Take the top remaining frames (since list_keyframe is already sorted)
            extra_frames = available_frames[:remaining_needed]
            selected_keyframe_list.extend(extra_frames)

        return selected_keyframe_list



    def get_pointcloud(self, depth, dataset, w2c, indices):
        CX = dataset.cx
        CY = dataset.cy
        FX = dataset.fx
        FY = dataset.fy

        # Compute indices of sampled pixels
        xx = (indices[:, 1] - CX) / FX
        yy = (indices[:, 0] - CY) / FY
        depth_z = depth[0, indices[:, 0], indices[:, 1]]

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)

        # w2c = getWorld2View2(viewpoint.R, viewpoint.T)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]

        # Remove points at camera origin
        A = torch.abs(torch.round(pts, decimals=4))
        B = torch.zeros((1, 3)).cuda().float()
        _, idx, counts = torch.cat([A, B], dim=0).unique(
            dim=0, return_inverse=True, return_counts=True)
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        invalid_pt_idx = mask[:len(A)]
        valid_pt_idx = ~invalid_pt_idx
        pts = pts[valid_pt_idx]

        return pts
    def compute_sampson_error(self,x1, x2, F):
        """
        :param x1 (*, N, 2)
        :param x2 (*, N, 2)
        :param F (*, 3, 3)
        """
        h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
        h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
        d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
        d2 = torch.matmul(h2, F)  # (B, N, 3)
        z = (h2 * d1).sum(dim=-1)  # (B, N)
        err = z ** 2 / (d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2)
        return err
    def get_uv_grid(self,H, W, homo=False, align_corners=False, device=None):
        """
        Get uv grid renormalized from -1 to 1
        :returns (H, W, 2) tensor
        """
        if device is None:
            device = torch.device("cpu")
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        if align_corners:
            xx = 2 * xx / (W - 1) - 1
            yy = 2 * yy / (H - 1) - 1
        else:
            xx = 2 * (xx + 0.5) / W - 1
            yy = 2 * (yy + 0.5) / H - 1
        if homo:
            return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        return torch.stack([xx, yy], dim=-1)

    def generate_flow(self, image, idx,image_last):
        if self.flow is not None:
            return self.flow, self.flow_back, self.mask_fwd, self.mask_bwd, self.epi_error
        args = raft_param()
        model = torch.nn.DataParallel(RAFT(args))
        # model = torch.nn.DataParallel(RAFTGMA(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            image_copy = image.detach().clone() * 255
            image_last_copy = image_last.detach().clone() * 255
            image_copy, image_last_copy = image_copy[None], image_last_copy[None]
            padder = InputPadder(image_last_copy.shape)
            image_last_copy, image_copy = padder.pad(image_last_copy, image_copy)

            _, flow_fwd = model(image_last_copy, image_copy, iters=20, test_mode=True)  # image_last->image
            _, flow_bwd = model(image_copy, image_last_copy, iters=20, test_mode=True)  # image->image_last

            # flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
            # flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

            flow_fwd = padder.unpad(flow_fwd[0]).permute(1, 2, 0)
            flow_bwd = padder.unpad(flow_bwd[0]).permute(1, 2, 0)

            flow_fwd_copy = flow_fwd.clone().cpu().numpy()
            flow_bwd_copy = flow_bwd.clone().cpu().numpy()
            flow_fimg = flow_viz.flow_to_image(flow_fwd_copy)
            flow_bimg = flow_viz.flow_to_image(flow_bwd_copy)
            # output_dir = os.path.join("output", "flow2_xyz_image")
            # os.makedirs(output_dir, exist_ok=True)
            # # image1 = image.permute(1, 2, 0).cpu().detach().numpy()
            # # image1_normalized = (image1 - image1.min()) / (
            # #         image1.max() - image1.min()) * 255
            # # depth_normalized = image1_normalized.astype(np.uint8)
            # # cv2_image = cv2.cvtColor(depth_normalized, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(output_dir, f"image1_flow{idx:4d}.png"), flow_fimg [:, :, [2, 1, 0]])
            mask_fwd, mask_bwd = self.compute_fwdbwd_mask(flow_fwd_copy, flow_bwd_copy)

            coor1to2_flow = flow_fwd / torch.tensor(flow_fwd.shape[:2][::-1], dtype=torch.float32).cuda() * 2
            coor1to2_flow_back = flow_bwd / torch.tensor(flow_bwd.shape[:2][::-1], dtype=torch.float32).cuda() * 2

        self.mask_fwd = torch.from_numpy(mask_fwd).float().cuda()
        self.mask_bwd = torch.from_numpy(mask_bwd).float().cuda()
        self.flow = coor1to2_flow
        self.flow_back = coor1to2_flow_back
        #self.epi_error = self.thread(idx, self.flow_back, self.mask_bwd, self.flow, self.mask_fwd,
                                #     np.ones((self.image_height, self.image_width)) > 0, self.image_height,
                              #       self.image_width)
        #print("epi",self.epi_error)
        return self.flow, self.flow_back, self.mask_fwd, self.mask_bwd,self.epi_error



    def warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow_new = flow.copy()
        flow_new[:, :, 0] += np.arange(w)
        flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

        res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return res

    def compute_fwdbwd_mask(self, fwd_flow, bwd_flow):
        alpha_1 = 0.5
        alpha_2 = 0.5

        bwd2fwd_flow = self.warp_flow(bwd_flow, fwd_flow)
        fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
        fwd_mask = fwd_lr_error < alpha_1 * (np.linalg.norm(fwd_flow, axis=-1) \
                                             + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

        fwd2bwd_flow = self.warp_flow(fwd_flow, bwd_flow)
        bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

        bwd_mask = bwd_lr_error < alpha_1 * (np.linalg.norm(bwd_flow, axis=-1) \
                                             + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

        return fwd_mask, bwd_mask

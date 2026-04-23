import torch


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False, rm_dynamic=False, mask=None):
    # 应用曝光补偿参数调整渲染图像
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    # 根据单目配置选择损失计算模式
    if config["Training"]["monocular"]:
        # 调用纯RGB损失计算（无深度约束）
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    # 调用RGB-D联合损失计算
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, rm_dynamic=rm_dynamic, mask=mask)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=False, mask=None):
    # 获取原始图像并转存到GPU
    gt_image = viewpoint.original_image.cuda()
    # 提取图像尺寸信息
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    # 生成RGB有效区域掩码（过滤过暗区域）
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]  # 从配置读取亮度阈值
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    # 结合梯度掩码（关注边缘区域）
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask

    # 动态物体掩码处理（当需要移除动态物体且非首帧时）
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid > 0:
        rgb_pixel_mask = viewpoint.motion_mask.view(*mask_shape) * rgb_pixel_mask

    # 应用外部传入的额外掩码（如重投影掩码）
    if mask is not None:
        rgb_pixel_mask = mask.view(*mask_shape) * rgb_pixel_mask

    # 计算加权L1损失（不透明度*RGB差异）
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()  # 返回平均损失值


def get_loss_tracking_rgbd(
        config, image, depth, opacity, viewpoint, initialization=False, rm_dynamic=False, mask=None
):
    # 设置RGB与深度损失的混合权重（默认0.95:0.05）
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    # 准备深度真值数据（转Tensor并匹配设备）
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    # 生成深度有效区域掩码（过滤过近/过远点）
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)  # 近裁剪面0.01米
    depth_pixel_mask *= (gt_depth < 1000.).view(*depth.shape)  # 远裁剪面1000米
    # 基于不透明度的可靠区域掩码
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    # 计算RGB分量损失
    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=rm_dynamic, mask=mask)
    # 合成最终深度掩码（深度有效区域∩高不透明度区域）
    depth_mask = depth_pixel_mask * opacity_mask

    # 动态物体掩码处理
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid > 0:
        depth_mask = viewpoint.motion_mask.view(*depth.shape) * depth_mask

    # 应用外部传入的额外掩码
    if mask is not None:
        depth_mask = mask.view(*depth.shape) * depth_mask

    # 计算深度L1损失
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    # 返回加权总损失（RGB权重α，深度权重1-α）
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_network_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=False, mask=None, dynamic=False):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (opacity>0.95).view(*mask_shape)
    #rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid>0:
        rgb_pixel_mask = viewpoint.motion_mask.view(*mask_shape) * rgb_pixel_mask
    if mask is not None and rm_dynamic:
        rgb_pixel_mask = mask.view(*mask_shape) * rgb_pixel_mask
    l1 = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    #######
    if dynamic:
        if mask is not None:
            l1[(~viewpoint.motion_mask.view(*mask_shape)).repeat(3, 1, 1) | (~mask.view(*mask_shape)).repeat(3, 1, 1)] *= 3
        else:
            l1[~viewpoint.motion_mask.view(*mask_shape).repeat(3, 1, 1)] *= 3
    #######
    return l1.mean()

def pearson_loss(depth, viewpoint):
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=depth.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape) 
    
    rendered_depth = (depth*depth_pixel_mask).view(-1)
    true_depth = (gt_depth*depth_pixel_mask).view(-1)

    mean_rendered = torch.mean(rendered_depth)
    mean_true = torch.mean(true_depth)

    numerator = torch.sum((rendered_depth - mean_rendered) * (true_depth - mean_true))
    denominator = torch.sqrt(torch.sum((rendered_depth - mean_rendered) ** 2) * torch.sum((true_depth - mean_true) ** 2))

    correlation = numerator / denominator

    loss = 1 - correlation

    return loss


def get_loss_network(config, image, depth, viewpoint, opacity, initialization=False, rm_dynamic=False, mask=None, alpha=None, dynamic=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if alpha is None:
        alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.9
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)
    gt_image = viewpoint.original_image.cuda()
    l1_rgb = get_loss_network_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=rm_dynamic, mask=mask, dynamic=dynamic)
    depth_mask = depth_pixel_mask * opacity_mask
    
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid>0:
        depth_mask = viewpoint.motion_mask.view(*depth.shape) * depth_mask
    if mask is not None and rm_dynamic:
        depth_mask = mask.view(*depth.shape) * depth_mask
        
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    #####
    if dynamic:
        if mask is not None:
            l1_depth[(~viewpoint.motion_mask.view(*depth.shape)) | (~mask.view(*depth.shape))] *= 3
        else:
            l1_depth[~viewpoint.motion_mask.view(*depth.shape)] *= 3
    #######
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()
    
    

def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, alpha=None, rm_dynamic=False, mask=None, dynamic=False, split=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, alpha=alpha, rm_dynamic=rm_dynamic, mask=mask, dynamic=dynamic, split=split)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, alpha=None, rm_dynamic=False, mask=None, dynamic=False, split=False):
    if alpha is None:
        alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    depth_pixel_mask *= (gt_depth < 10000.).view(*depth.shape)
    
    if viewpoint.motion_mask is not None and rm_dynamic:  # and viewpoint.uid>0:
        rgb_pixel_mask = viewpoint.motion_mask.view(*depth.shape) * rgb_pixel_mask
        depth_pixel_mask = viewpoint.motion_mask.view(*depth.shape) * depth_pixel_mask
    if mask is not None and rm_dynamic:
        rgb_pixel_mask = mask.view(*depth.shape) * rgb_pixel_mask
        depth_pixel_mask = mask.view(*depth.shape) * depth_pixel_mask

    if split:
        motion_mask = viewpoint.motion_mask.view(*depth.shape)
        l1_static = alpha * torch.abs(motion_mask * rgb_pixel_mask*image-motion_mask * rgb_pixel_mask*gt_image).mean()
        l1_static += (1-alpha) * torch.abs(motion_mask * depth_pixel_mask*depth-motion_mask * depth_pixel_mask*gt_depth).mean()
        
        l1_dynamic = alpha * torch.abs((~motion_mask) * rgb_pixel_mask*image-(~motion_mask) * rgb_pixel_mask*gt_image).mean()
        l1_dynamic += (1-alpha) * torch.abs((~motion_mask) * depth_pixel_mask*depth-(~motion_mask) * depth_pixel_mask*gt_depth).mean()
        if dynamic:
            return l1_static, 2*l1_dynamic
        else:
            return l1_static, l1_dynamic
        
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    ######
    if dynamic:
        if mask is not None:
            #l1_depth[(~viewpoint.motion_mask.view(*depth.shape)) | (~mask.view(*depth.shape))] *= 5
            #l1_rgb[((~viewpoint.motion_mask.view(*depth.shape)) | (~mask.view(*depth.shape))).repeat(3, 1, 1)] *= 5
            l1_depth[mask.view(*depth.shape).bool().detach()|~viewpoint.motion_mask.view(*depth.shape).detach()] *= 2
            l1_rgb[mask.view(*depth.shape).repeat(3, 1, 1).bool().detach()|~viewpoint.motion_mask.view(*depth.shape).repeat(3, 1, 1).detach()] *= 2
            #l1_rgb[((~viewpoint.motion_mask.view(*depth.shape)) | (mask.view(*depth.shape))).repeat(3, 1, 1)] *= 3 ����ã���
        else:
            l1_depth[~viewpoint.motion_mask.view(*depth.shape)] *= 2
            l1_rgb[~viewpoint.motion_mask.view(*depth.shape).repeat(3, 1, 1)] *= 2
    ######
    
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

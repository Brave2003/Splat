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


import numpy as np
from lietorch import SE3
from src.utils.Printer import FontColor

def align_kf_traj(npz_path,stream,return_full_est_traj=False,printer=None):
    offline_video = dict(np.load(npz_path))
    traj_ref = []
    traj_est = []
    video_traj = offline_video['poses']
    video_timestamps = offline_video['timestamps']
    timestamps = []

    for i in range(video_timestamps.shape[0]):
        timestamp = int(video_timestamps[i])
        val = stream.poses[timestamp].sum()
        if np.isnan(val) or np.isinf(val):
            printer.print(f'Nan or Inf found in gt poses, skipping {i}th pose!',FontColor.INFO)
            continue
        traj_est.append(video_traj[i])
        traj_ref.append(stream.poses[timestamp])
        timestamps.append(video_timestamps[i])

    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

    if return_full_est_traj:
        from evo.core import lie_algebra as lie
        traj_est_full = PoseTrajectory3D(poses_se3=video_traj,timestamps=video_timestamps)
        traj_est_full.scale(s)
        traj_est_full.transform(lie.se3(r_a, t_a))
        traj_est = traj_est_full

    return r_a, t_a, s, traj_est, traj_ref    

def align_full_traj(traj_est_full,stream,printer):

    timestamps = []
    traj_ref = []
    traj_est = []
    for i in range(len(stream.poses)):
        val = stream.poses[i].sum()
        if np.isnan(val) or np.isinf(val):
            printer.print(f'Nan or Inf found in gt poses, skipping {i}th pose!',FontColor.INFO)
            continue
        traj_est.append(traj_est_full[i])
        traj_ref.append(stream.poses[i])
        timestamps.append(float(i))
    
    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)
    return r_a, t_a, s, traj_est, traj_ref


def traj_eval_and_plot(traj_est, traj_ref, plot_parent_dir, plot_name, printer):
    # 导入操作系统接口模块
    import os
    # 从evo库导入SLAM评估指标模块
    from evo.core import metrics
    # 从evo工具库导入绘图模块
    #from evo.tools import plot
    # 导入matplotlib绘图库
    import matplotlib
    # 强制使用非GUI的'Agg'后端(解决服务器无图形界面问题)
    matplotlib.use('Agg')
    # 导入pyplot绘图接口
    import matplotlib.pyplot as plt

    # 检查输出目录是否存在
    if not os.path.exists(plot_parent_dir):
        # 递归创建多级目录
        os.makedirs(plot_parent_dir)

        # 打印评估开始提示(使用自定义字体颜色)
    printer.print("Calculating APE ...", FontColor.EVAL)

    # 组织轨迹数据对
    data = (traj_ref, traj_est)
    # 创建绝对位姿误差(APE)度量对象，专注平移部分
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    # 处理输入轨迹数据
    ape_metric.process_data(data)
    # 获取APE统计结果(均值、标准差等)
    ape_statistics = ape_metric.get_all_statistics()

    # # 打印绘图开始提示
    # printer.print("Plotting ...", FontColor.EVAL)
    # # 关闭交互模式，防止图形窗口弹出
    # plt.ioff()
    #
    # # 创建绘图集合对象(用于管理多个图表)
    # plot_collection = plot.PlotCollection("kf factor graph")
    # print("picture")
    # # 创建8x8英寸的画布
    # fig_1 = plt.figure(figsize=(8, 8))
    # # 设置二维XY绘图模式
    # plot_mode = plot.PlotMode.xy
    # # 准备坐标轴对象
    # ax = plot.prepare_axis(fig_1, plot_mode)
    #
    # # 绘制参考轨迹(灰色虚线)
    # plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
    # # 绘制估计轨迹并用颜色映射APE误差
    # plot.traj_colormap(
    #     ax,  # 目标坐标轴
    #     traj_est,  # 待评估轨迹
    #     ape_metric.error,  # 误差数据
    #     plot_mode,  # 绘图模式
    #     min_map=ape_statistics["min"],  # 颜色映射最小值
    #     max_map=ape_statistics["max"],  # 颜色映射最大值
    #     title="APE mapped onto trajectory"  # 图表标题
    # )
    #
    # # 将图表添加到集合的"2d"分类
    # plot_collection.add_figure("2d", fig_1)
    #
    # # 构建输出文件路径
    # output_path = os.path.join(plot_parent_dir, f"{plot_name}.png")
    # # 保存图表到文件(不弹出覆盖确认)
    # plot_collection.export(output_path, confirm_overwrite=False)
    #
    # # 关闭所有图形对象释放内存
    # plt.close('all')

    # 返回APE统计结果
    return ape_statistics


def kf_traj_eval(npz_path, plot_parent_dir,plot_name, stream, logger,printer):
    r_a, t_a, s, traj_est, traj_ref = align_kf_traj(npz_path, stream,printer=printer)

    offline_video = dict(np.load(npz_path))
    
    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name,printer)

    output_str = "#"*10+"Keyframes traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}"
    printer.print(output_str,FontColor.EVAL)
    printer.print("#"*34,FontColor.EVAL)
    out_path=f'{plot_parent_dir}/metrics_kf_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'kf_ate_rmse':ape_statistics['rmse'],'pose_scale':s})

    offline_video["scale"]=np.array(s)
    np.savez(npz_path,**offline_video)

    return ape_statistics, s, r_a, t_a


def full_traj_eval(traj_filler, plot_parent_dir, plot_name, stream,logger,printer):

    traj_est_inv = traj_filler(stream)
    traj_est_lietorch = traj_est_inv.inv()
    traj_est = traj_est_lietorch.matrix().data.cpu().numpy()
    kf_num = traj_filler.video.counter.value
    kf_timestamps = traj_filler.video.timestamp[:kf_num].cpu().int().numpy()
    kf_poses = SE3(traj_filler.video.poses[:kf_num].clone()).inv().matrix().data.cpu().numpy()
    traj_est[kf_timestamps] = kf_poses
    traj_est_not_align = traj_est.copy()

    r_a, t_a, s, traj_est, traj_ref = align_full_traj(traj_est, stream, printer)    

    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name,printer)
    output_str = "#"*10+"Full traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}"
    printer.print(output_str,FontColor.EVAL)
    printer.print("#"*29,FontColor.EVAL)

    
    out_path=f'{plot_parent_dir}/metrics_full_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'full_ate_rmse':ape_statistics['rmse']})
    return traj_est_not_align, traj_est, traj_ref

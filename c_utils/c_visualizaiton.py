import os
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plt_fig(test_imgs, scores, img_scores, gt_mask_list, seg_threshold, cls_threshold, save_dir, class_name):
    """
    可视化异常检测模型的测试结果，生成包含输入图像、真实掩码、热力图与分类信息的综合图像。

    对每张测试图像生成一个 4 子图的可视化结果：
    1. Input image: 原始输入图像
    2. GroundTruth: 真实缺陷掩码（人工标注）
    3. Segmentation: 异常热力图 + 检测边界叠加
    4. Classification: 分类结果与阈值信息面板

    结果以高分辨率保存至指定目录。

    Args:
        test_imgs (list or np.ndarray): 测试图像列表，每个图像为 (C, H, W) 格式，已归一化。
        scores (np.ndarray): 像素级异常分数，形状为 (N, H, W)，值范围 [0, 1]。
        img_scores (list or np.ndarray): 图像级异常分数，长度为 N，用于整体分类判断。
        gt_mask_list (list or np.ndarray): 真实缺陷掩码列表，每个为 (H, W) 的二值图。
        seg_threshold (float): 像素级分割阈值，用于生成异常区域掩码。
        cls_threshold (float): 图像级分类阈值，用于判断整张图是否异常。
        save_dir (str): 可视化结果保存的根目录路径。
        class_name (str): 数据集类别名称（如 'bottle', 'capsule'），用于文件命名。

    Returns:
        None: 直接将可视化图像保存到磁盘，不返回任何值。

    Example:
        plt_fig(
            test_imgs=test_images,
            scores=pixel_anomaly_scores,
            img_scores=image_level_scores,
            gt_mask_list=ground_truths,
            seg_threshold=0.5,
            cls_threshold=0.6,
            save_dir='./results',
            class_name='carpet'
        )
        # 保存文件：./results/carpet_0.png, ./results/carpet_1.png, ...
    """
    # 调整分割阈值：略微降低，使更多潜在异常区域被检出（减少漏报）
    threshold = seg_threshold - 0.1
    num = len(scores)  # 测试样本数量

    # 计算热力图颜色映射范围：将分数缩放到 [0, 255]
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    # 进一步压缩最大值，增强低分区域的视觉对比度
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # 遍历每个测试样本
    for i in range(num):
        img = test_imgs[i]
        img = denormalization(img)  # 将归一化的图像还原为 [0, 255] 范围
        gt = gt_mask_list[i].squeeze()  # 获取对应的真实掩码，并去除冗余维度
        heat_map = scores[i] * 255  # 将像素分数转为热力图强度
        if heat_map.ndim == 3:
            heat_map = heat_map.squeeze()
        mask = scores[i].copy()  # 复制分数图用于生成二值分割掩码

        if mask.ndim == 3:
            mask = mask.squeeze()

        # 二值化：高于阈值设为1（异常），否则为0（正常）
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0

        # 形态学开运算：使用半径为4的圆形结构元素，去除小噪点并平滑边缘
        kernel = morphology.disk(4)
        mask = morphology.opening(mask.astype(np.float64), kernel)
        mask *= 255  # 缩放到 [0, 255] 以便可视化

        # 在原始图像上标记异常区域边界（红色粗线）
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        # 创建 1x4 子图布局，设置宽高比和间距
        fig_img, ax_img = plt.subplots(1, 3, figsize=(7, 3), gridspec_kw={'width_ratios': [4, 4, 4]})
        fig_img.subplots_adjust(wspace=0.05, hspace=0)  # 紧凑布局

        # 隐藏所有子图的坐标轴
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        # 子图 0: 输入图像
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Input image')

        # 子图 1: 真实缺陷掩码
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')

        # 子图 2: 异常热力图 + 边界叠加
        ax_img[2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')  # jet 热力图
        ax_img[2].imshow(vis_img, cmap='gray', alpha=0.7, interpolation='none')  # 红色边界
        ax_img[2].imshow(img, cmap='gray', alpha=0.1, interpolation='none')      # 原图背景
        ax_img[2].title.set_text('Segmentation')

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 保存可视化图像，高分辨率，裁剪空白边缘
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)),
                        dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图像防止内存泄漏

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x
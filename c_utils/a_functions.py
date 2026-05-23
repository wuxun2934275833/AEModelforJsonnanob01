import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
'''
def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 0.5 * (ft_norm - fs_norm)**2
        f_loss = f_loss.sum() / (h*w)
        t_loss += f_loss
    return t_loss / N

def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape

        a_map = (0.5 * (ft_norm - fs_norm)**2) / (h*w)

        a_map = a_map.sum(1, keepdim=True)

        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        anomaly_map += a_map
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
    return anomaly_map


from skimage.measure import label, regionprops


def compute_region_f1(gt_mask, pred_mask, iou_thresh=0.3, overlap_ratio_thresh=0.5):
    """
    计算单张图像的区域级 F1 分数 (Region-level F1 Score)。

    参数:
        gt_mask (np.ndarray): 真实标签掩码，形状 (H, W)，二值 (0/1)
        pred_mask (np.ndarray): 预测掩码，形状 (H, W)，二值 (0/1)
        iou_thresh (float): IoU 匹配阈值，默认 0.3
        overlap_ratio_thresh (float): 重叠面积比阈值，默认 0.5

    返回:
        f1_reg (float): 区域级 F1 分数 [0, 1]
        tp (int): True Positives (成功匹配的预测区域数)
        fp (int): False Positives (未匹配的预测区域数)
        fn (int): False Negatives (未被匹配的真实区域数)
    """
    # 确保输入是二值的布尔或整数数组
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    # 提取连通区域（标记每个独立区域）
    gt_labeled = label(gt_mask, connectivity=2)  # 8-邻域
    pred_labeled = label(pred_mask, connectivity=2)

    # 获取区域数量
    num_gt = gt_labeled.max()
    num_pred = pred_labeled.max()

    # 特殊情况：无真实缺陷且无预测 → 完美，F1=1
    if num_gt == 0 and num_pred == 0:
        return 1.0, 0, 0, 0
    # 无真实缺陷但有预测 → 全是误报
    if num_gt == 0:
        return 0.0, 0, num_pred, 0
    # 有真实缺陷但无预测 → 全漏检
    if num_pred == 0:
        return 0.0, 0, 0, num_gt

    # 初始化匹配状态
    matched_gt = set()
    tp = 0

    # 遍历每个预测区域
    for pred_id in range(1, num_pred + 1):
        pred_region = (pred_labeled == pred_id)
        matched = False

        # 尝试与每个未匹配的真实区域匹配
        for gt_id in range(1, num_gt + 1):
            if gt_id in matched_gt:
                continue
            gt_region = (gt_labeled == gt_id)

            # 计算交集
            intersection = np.logical_and(pred_region, gt_region)
            inter_area = np.sum(intersection)
            if inter_area == 0:
                continue

            pred_area = np.sum(pred_region)
            gt_area = np.sum(gt_region)
            union_area = pred_area + gt_area - inter_area

            # 计算 IoU
            iou = inter_area / union_area

            # 计算重叠面积比（ARR）
            overlap_pred = inter_area / pred_area
            overlap_gt = inter_area / gt_area

            # 检查匹配条件（满足任一即可）
            if (iou > iou_thresh) or (overlap_pred > overlap_ratio_thresh) or (overlap_gt > overlap_ratio_thresh):
                # 成功匹配
                matched_gt.add(gt_id)
                tp += 1
                matched = True
                break  # 一个预测区域只匹配一个真实区域

        # 如果没匹配上，就是 FP（但这里不计数，最后统一算）

    fp = num_pred - tp
    fn = num_gt - tp

    # 计算 F1
    if tp == 0:
        f1_reg = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_reg = 2 * precision * recall / (precision + recall)

    return f1_reg, tp, fp, fn


# ---------------------------
# 批量计算整个测试集的区域级 F1（micro-F1）
# ---------------------------
def compute_region_f1_dataset(gt_masks, pred_masks, **kwargs):
    """
    对整个数据集计算区域级 F1（汇总所有图像的 TP/FP/FN 后计算）

    参数:
        gt_masks (list of np.ndarray): 所有真实掩码列表
        pred_masks (list of np.ndarray): 所有预测掩码列表
        **kwargs: 传递给 compute_region_f1 的参数

    返回:
        f1_reg (float): 整体区域级 F1 分数
    """
    total_tp = total_fp = total_fn = 0

    for gt, pred in zip(gt_masks, pred_masks):
        _, tp, fp, fn = compute_region_f1(gt, pred, **kwargs)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp == 0:
        return 0.0
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1_reg = 2 * precision * recall / (precision + recall)
    return f1_reg'''
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.measure import label


def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 0.5 * (ft_norm - fs_norm)**2
        f_loss = f_loss.sum() / (h * w)
        t_loss += f_loss
    return t_loss / N


def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape

        a_map = (0.5 * (ft_norm - fs_norm)**2) / (h * w)
        a_map = a_map.sum(1, keepdim=True)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        anomaly_map += a_map

    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    if anomaly_map.ndim == 2:
        anomaly_map = anomaly_map[None, ...]  # add batch dim if single image
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
    return anomaly_map


def compute_region_f1(gt_mask, pred_mask, iou_thresh=0.3, overlap_ratio_thresh=0.5):
    """
    计算单张图像的区域级 F1 分数 (Region-level F1 Score)。
    """
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    gt_labeled = label(gt_mask, connectivity=2)
    pred_labeled = label(pred_mask, connectivity=2)

    num_gt = gt_labeled.max()
    num_pred = pred_labeled.max()

    if num_gt == 0 and num_pred == 0:
        return 1.0, 0, 0, 0
    if num_gt == 0:
        return 0.0, 0, num_pred, 0
    if num_pred == 0:
        return 0.0, 0, 0, num_gt

    matched_gt = set()
    tp = 0

    for pred_id in range(1, num_pred + 1):
        pred_region = (pred_labeled == pred_id)
        for gt_id in range(1, num_gt + 1):
            if gt_id in matched_gt:
                continue
            gt_region = (gt_labeled == gt_id)

            intersection = np.logical_and(pred_region, gt_region)
            inter_area = np.sum(intersection)
            if inter_area == 0:
                continue

            pred_area = np.sum(pred_region)
            gt_area = np.sum(gt_region)
            union_area = pred_area + gt_area - inter_area

            iou = inter_area / union_area
            overlap_pred = inter_area / pred_area
            overlap_gt = inter_area / gt_area

            if (iou > iou_thresh) or (overlap_pred > overlap_ratio_thresh) or (overlap_gt > overlap_ratio_thresh):
                matched_gt.add(gt_id)
                tp += 1
                break

    fp = num_pred - tp
    fn = num_gt - tp

    if tp == 0:
        f1_reg = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_reg = 2 * precision * recall / (precision + recall)

    return f1_reg, tp, fp, fn


def compute_region_f1_dataset(
    ground_truth_masks,
    preds_scores,
    threshold=0.5,
    iou_thresh=0.3,
    overlap_ratio_thresh=0.5
):
    """
    对整个测试集计算区域级 F1（micro-F1），输入为真实掩码和预测分数图。

    参数:
        ground_truth_masks (list of np.ndarray): 真实标签掩码列表，每个形状 (H, W)，二值 (0/1)
        preds_scores (list of np.ndarray): 预测分数图（如 anomaly map），每个形状 (H, W)
        threshold (float): 二值化阈值，默认 0.5
        iou_thresh (float): IoU 匹配阈值，默认 0.3
        overlap_ratio_thresh (float): 重叠面积比阈值，默认 0.5

    返回:
        f1_reg (float): 整体区域级 F1 分数（micro-F1）
    """
    total_tp = total_fp = total_fn = 0

    for gt_mask, pred_score in zip(ground_truth_masks, preds_scores):
        # 二值化预测分数图
        pred_mask = (pred_score >= threshold).astype(np.uint8)
        # 计算单图区域匹配结果
        _, tp, fp, fn = compute_region_f1(
            gt_mask,
            pred_mask,
            iou_thresh=iou_thresh,
            overlap_ratio_thresh=overlap_ratio_thresh
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp == 0:
        return 0.0
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1_reg = 2 * precision * recall / (precision + recall)
    return f1_reg



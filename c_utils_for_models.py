from sklearn.metrics import f1_score
import swanlab
import numpy as np
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from c_utils.c_visualizaiton import plt_fig
from c_utils.a_functions import compute_region_f1_dataset
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score  # 注意：sklearn 不支持 GPU，所以我们要自己实现 F1！
from c_utils.a_functions import compute_region_f1_dataset



def calculate_best_sample_f1(scores,gt_list):#this if for sure correct
    #calculate sample f1
    print("scores and gtlist:",np.isnan(scores).any(),np.isnan(gt_list).any())
    # # 在空间维度 (w, h) 上求 min/max，保持维度以便广播
    # min_vals = scores.min(axis=(2, 3), keepdims=True)  # shape: (b, c, 1, 1)
    # max_vals = scores.max(axis=(2, 3), keepdims=True)  # shape: (b, c, 1, 1)
    # denom = max_vals - min_vals
    # # 利用广播自动扩展 (b,c,1,1) 到 (b,c,w,h)
    # normalized_scores = (scores - min_vals) / denom
    # scores = normalized_scores
    # do not do normalization process
    img_scores_ravel = scores.reshape(scores.shape[0], -1).max(axis=1)
    img_gt_ravel=np.array(gt_list).ravel()
    precisions, recalls, thresholds = precision_recall_curve(img_gt_ravel, img_scores_ravel, pos_label=1)
    precisions = precisions[:-1]#we used slice because the last precision and recall is 0
    recalls = recalls[:-1]

    # 计算 F1 (Dice)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    # 找到最大 F1 对应的索引
    max_idx = np.nanargmax(f1_scores)

    best_threshold = thresholds[max_idx]
    best_f1 = f1_scores[max_idx]

    return best_threshold, best_f1


def calculate_best_sample_f1_cuda(scores, gt_list):
    """
    Calculate best sample-level F1 score with optional GPU acceleration.

    Parameters:
        scores (np.ndarray or torch.Tensor):
            Anomaly score maps, shape (B, C, H, W). Can be on CPU or GPU.
        gt_list (array-like):
            Binary image-level labels (0=normal, 1=anomaly), length = B.

    Returns:
        best_threshold (float): Optimal threshold for max F1.
        best_f1 (float): Maximum F1 score.
    """
    # --- Step 1: 确保输入是 PyTorch Tensor ---
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores).float()
    elif not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be np.ndarray or torch.Tensor")

    # --- Step 2: 在 GPU/CPU 上进行归一化和图像级分数提取 ---
    with torch.no_grad():  # 禁用梯度，节省内存
        # 归一化：沿空间维度 (H, W) 进行 min-max
        # min_vals = scores.amin(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        # max_vals = scores.amax(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        # denom = max_vals - min_vals
        #
        # # 防止除零：当 max == min 时，设 denom = 1（此时 normalized_score = 0）
        # denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        # normalized_scores = (scores - min_vals) / denom

        # 每张图取全局最大异常分数（跨 C, H, W）
        img_scores = scores.flatten(start_dim=1).amax(dim=1)  # (B,)

    # --- Step 3: 转回 CPU NumPy，供 sklearn 使用 ---
    img_scores_cpu = img_scores.cpu().numpy()
    img_gt_ravel = np.array(gt_list).ravel()

    # 安全检查
    if len(img_scores_cpu) != len(img_gt_ravel):
        raise ValueError(f"Batch size mismatch: scores has {len(img_scores_cpu)} samples, "
                         f"but gt_list has {len(img_gt_ravel)} labels.")

    # --- Step 4: 使用 sklearn 计算 PR curve 和 F1 ---
    precisions, recalls, thresholds = precision_recall_curve(
        img_gt_ravel, img_scores_cpu, pos_label=1
    )
    print(f"the maximun sample threshold seen is:{thresholds.max()}")
    print(f"the minimun sample threshold seen is:{thresholds.min()}")
    swanlab.log({"maximun samplef1 threshold":thresholds.max()})
    swanlab.log({"minimun samplef1 threshold":thresholds.min()})
    # 移除最后一个点（sklearn 添加的 (1, 0) 点）
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # 计算 F1
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    # 找最大 F1（处理全 NaN 情况）
    if np.all(np.isnan(f1_scores)):
        return 0.0, 0.0

    max_idx = np.nanargmax(f1_scores)
    best_threshold = float(thresholds[max_idx])
    best_f1 = float(f1_scores[max_idx])

    return best_threshold, best_f1
def calculate_best_pixel_f1(scores,gt_list):
    #calculate sample f1
    print("scores and gtlist:",np.isnan(scores).any(),np.isnan(gt_list).any())

    #calculate pixel f1
    img_scores_ravel = np.array(scores).ravel()
    img_gt_ravel=np.array(gt_list).ravel()
    precisions, recalls, thresholds = precision_recall_curve(img_gt_ravel, img_scores_ravel, pos_label=1)
    print(f"the maximun pixelf1 threshold seen is:{thresholds.max()}")
    print(f"the minimun pixelf1 threshold seen is:{thresholds.min()}")
    swanlab.log({"maximun pixelf1 threshold":thresholds.max()})
    swanlab.log({"minimun pixelf1 threshold":thresholds.min()})
    precisions = precisions[:-1]#we used slice because the last precision and recall is 0
    recalls = recalls[:-1]

    # 计算 F1 (Dice)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    # 找到最大 F1 对应的索引
    max_idx = np.nanargmax(f1_scores)

    best_threshold = thresholds[max_idx]
    best_f1 = f1_scores[max_idx]

    return best_threshold, best_f1

from scipy.ndimage import label, find_objects


def _compute_regional_f1_single(pred_mask, gt_mask, iou_threshold=0.3):
    """
    计算单张图像的 regional F1（基于连通区域匹配）。

    定义（参考 ZL 挑战赛）：
      - 将 pred 和 gt 分别做连通域分析
      - 对每个 pred 区域，若与任一 gt 区域 IoU >= iou_threshold，则视为 TP
      - FP = pred 区域数 - TP
      - FN = gt 区域数 - TP
      - F1 = 2*TP / (2*TP + FP + FN)
    """
    if pred_mask.ndim != 2 or gt_mask.ndim != 2:
        raise ValueError("Input masks must be 2D for single-image mode.")

    # 连通域标记（4-连通）
    pred_labels, n_pred = label(pred_mask.astype(int), structure=np.ones((3, 3)))
    gt_labels, n_gt = label(gt_mask.astype(int), structure=np.ones((3, 3)))

    if n_pred == 0 and n_gt == 0:
        return 1.0  # 完美：无缺陷且无误报
    if n_pred == 0 or n_gt == 0:
        return 0.0  # 一方为空，F1=0

    # 获取所有区域 bounding box（加速 IoU 计算）
    pred_slices = find_objects(pred_labels)
    gt_slices = find_objects(gt_labels)

    tp = 0
    matched_gt = set()

    for i in range(n_pred):
        if pred_slices[i] is None:
            continue
        pred_region = (pred_labels == (i + 1))
        pred_box = pred_slices[i]
        pred_crop = pred_region[pred_box]

        best_iou = 0.0
        best_j = -1

        for j in range(n_gt):
            if j in matched_gt:
                continue
            if gt_slices[j] is None:
                continue
            gt_region = (gt_labels == (j + 1))
            gt_box = gt_slices[j]

            # 计算交集区域（bounding box 交集）
            y_min = max(pred_box[0].start, gt_box[0].start)
            y_max = min(pred_box[0].stop, gt_box[0].stop)
            x_min = max(pred_box[1].start, gt_box[1].start)
            x_max = min(pred_box[1].stop, gt_box[1].stop)

            if y_max <= y_min or x_max <= x_min:
                iou = 0.0
            else:
                # 提取重叠区域
                pred_overlap = pred_crop[
                    y_min - pred_box[0].start: y_max - pred_box[0].start,
                    x_min - pred_box[1].start: x_max - pred_box[1].start
                ]
                gt_crop = gt_region[gt_box]
                gt_overlap = gt_crop[
                    y_min - gt_box[0].start: y_max - gt_box[0].start,
                    x_min - gt_box[1].start: x_max - gt_box[1].start
                ]

                intersection = np.logical_and(pred_overlap, gt_overlap).sum()
                union = pred_overlap.sum() + gt_overlap.sum() - intersection
                iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_j)

    fp = n_pred - tp
    fn = n_gt - tp
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return f1


def calculate_best_regional_f1(score_map, binary_target, thresholds=None, iou_threshold=0.3):
    """
    计算在所有候选阈值中能达到的最大 regional F1 分数（不依赖任何外部评估函数）。
    Parameters
    ----------
    score_map : np.ndarray
        异常评分图，shape (H, W) 或 (B, H, W)
    binary_target : np.ndarray
        真实缺陷掩码（bool），与 score_map 同 shape
    thresholds : array-like or None
        候选阈值。若为 None，则自动选择（最多 200 个）
    iou_threshold : float
        区域匹配所需的最小 IoU（默认 0.2，符合 ZL 标准）

    Returns
    -------
    best_threshold : float
        最优阈值
    max_regional_f1 : float
        对应的最大 regional F1
    """
    score_map = np.asarray(score_map)
    binary_target = np.asarray(binary_target).astype(bool)

    print("scores and gtlist:",np.isnan(score_map).any(),np.isnan(binary_target).any())

    if score_map.shape != binary_target.shape:
        raise ValueError("score_map and binary_target must have the same shape.")

    # 自动生成阈值
    if thresholds is None:
        unique_vals = np.unique(score_map)

        if len(unique_vals) > 200:
            thresholds = np.linspace(score_map.min(), score_map.max(), 200)
            print(f"the maximun region1 threshold seen is:{thresholds.max()}")
            print(f"the minimun region1 threshold seen is:{thresholds.min()}")
            swanlab.log({"maximun regionf1 threshold": thresholds.max()})
            swanlab.log({"minimun region1 threshold": thresholds.min()})
        else:
            thresholds = unique_vals

    # 处理 batch 维度
    if score_map.ndim == 2:
        score_map = score_map[None, ...]
        binary_target = binary_target[None, ...]

    best_f1 = -1.0
    best_thr = thresholds[0]

    for thr in thresholds:
        binary_pred = (score_map > thr)
        f1_scores = []
        binary_pred,binary_target=binary_pred,binary_target.squeeze()
        for i in range(binary_pred.shape[0]):
            f1 = _compute_regional_f1_single(
                binary_pred[i].squeeze(), binary_target[i].squeeze(), iou_threshold=iou_threshold
            )
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thr = thr

    return best_thr, best_f1



def calculScoreAndVisualize(model, scores, gt_list, gt_mask_list, test_imgs):
    """
    对模型输出的异常分数进行评估。
    只计算图像级ROCAUC和像素级ROCAUC。
    Args:
        model: 训练好的模型对象，包含配置信息（如是否可视化、保存路径等）
        scores (np.ndarray): 像素级异常分数，形状为 (N, H, W)，N为图像数量
        gt_list (list): 图像级标签列表，0表示正常，1表示异常
        gt_mask_list (list): 像素级标注掩码列表，每个元素为 (H, W) 的二值图
        test_imgs (list): 测试图像列表，用于可视化叠加显示

    Returns:
        None. 结果通过打印和保存图像输出。
    """
    # Step 1: 将异常分数归一化到 [0, 1] 范围
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # Step 2: 提取图像级异常分数 —— 每张图取所有像素中的最大值作为整体异常程度
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  # shape: (N,)
    gt_list = np.asarray(gt_list)

    # Step 3: 计算图像级分类性能：ROCAUC
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    swanlab.log({"img_roc_auc": img_roc_auc})

    # Step 4: 计算像素级分割性能：Pixel-level ROCAUC
    gt_mask_flat = np.asarray(gt_mask_list).flatten()
    pred_score_flat = scores.flatten()

    if np.unique(gt_mask_flat).size > 1:
        pixel_roc_auc = roc_auc_score(gt_mask_flat, pred_score_flat)
        print('pixel ROCAUC: %.3f' % (pixel_roc_auc))
        swanlab.log({"pixel_roc_auc": pixel_roc_auc})
    else:
        print("Pixel-level ROCAUC not defined (only one class in ground truth).")
        pixel_roc_auc = None

    if getattr(model, 'vis', False):
        save_dir = model.img_dir
        os.makedirs(save_dir, exist_ok=True)
        cls_threshold = 0.5
        seg_threshold = 0.5
        plt_fig(test_imgs, scores, img_scores, gt_mask_list, seg_threshold, cls_threshold, save_dir, model.group)
        print(f"Visualization saved to: {save_dir}")

    return img_roc_auc, pixel_roc_auc
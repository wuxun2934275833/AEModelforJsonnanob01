"""
Jetson Nano 实时异常检测推理脚本
用法: python3 realtime_inference.py --model_path ./results/models/10_gpu0_repeat1/model_s.pth --pattern 10
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import yaml
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from b_models_and_sub_models.a_teacher_model import TeacherModel
from b_models_and_sub_models.b_student_model import reverse_student18, ReverseStudent, BasicBlockDe
from c_utils.a_functions import cal_anomaly_maps, cal_loss


def load_models(model_path, device):
    """加载教师模型和学生模型"""
    # 加载教师模型 (用于提取特征)
    model_t = TeacherModel(backbone_name="resnet18", out_indices=[0, 1, 2, 3]).to(device)
    for param in model_t.parameters():
        param.requires_grad = False
    model_t.eval()

    # 加载学生模型
    model_s = reverse_student18(DG=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_s.load_state_dict(checkpoint['model'])
    model_s.eval()

    return model_t, model_s


class RealtimeInference:
    def __init__(self, model_path, pattern=10, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载模型
        self.model_t, self.model_s = load_models(model_path, self.device)
        self.pattern = pattern
        self.img_size = 256

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, img):
        """预处理图像"""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.transform(img).unsqueeze(0).to(self.device)

    def infer(self, img):
        """执行推理"""
        with torch.no_grad():
            # 数据预处理
            img_tensor = self.preprocess(img)

            # 教师模型提取特征
            features_t = self.model_t(img_tensor)

            # 学生模型计算异常图
            anomaly_map = cal_anomaly_maps(self.model_s(features_t), features_t, self.img_size)

            return anomaly_map

    def run_camera(self, camera_id=0, threshold=None):
        """运行实时摄像头推理"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return

        print("按 'q' 键退出，按 's' 键保存当前帧")

        fps_list = []
        last_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break

            # 执行推理
            anomaly_map = self.infer(frame)

            # 计算FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if last_time != current_time else 0
            fps_list.append(fps)
            last_time = current_time
            avg_fps = np.mean(fps_list[-30:]) if len(fps_list) > 30 else fps

            # 处理异常图
            result_frame, gray_map = self.postprocess_map(anomaly_map[0], frame)

            # 如果没有设置阈值，自动计算
            if threshold is None:
                threshold = np.mean(gray_map) + 2 * np.std(gray_map)

            # 标记异常区域
            mask = gray_map > threshold
            result_frame[mask] = [0, 0, 255]  # 红色标记异常区域

            # 显示信息
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Pattern: {self.pattern}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示异常分数
            anomaly_score = np.max(gray_map)
            cv2.putText(result_frame, f"Score: {anomaly_score:.2f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示画面
            cv2.imshow('Real-time Anomaly Detection', result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'anomaly_capture_{int(time.time())}.jpg'
                cv2.imwrite(filename, result_frame)
                print(f"已保存: {filename}")

        cap.release()
        cv2.destroyAllWindows()

    def postprocess_map(self, anomaly_map, original_img):
        """将异常图处理为可视化结果"""
        # 缩放到原始图像尺寸
        anomaly_map = cv2.resize(anomaly_map, (original_img.shape[1], original_img.shape[0]))
        # 归一化到0-255
        anomaly_map = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        # 应用颜色映射
        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        # 叠加到原图
        result = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        return result, anomaly_map

    def run_video(self, video_path, output_path=None):
        """处理视频文件"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"视频信息: {width}x{height} @ {fps}fps")
        print("按 'q' 键退出")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 执行推理
            anomaly_map = self.infer(frame)
            result_frame, gray_map = self.postprocess_map(anomaly_map[0], frame)

            # 标记异常区域
            threshold = np.mean(gray_map) + 2 * np.std(gray_map)
            mask = gray_map > threshold
            result_frame[mask] = [0, 0, 255]

            # 添加信息
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if output_path:
                out.write(result_frame)

            cv2.imshow('Video Anomaly Detection', result_frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed = time.time() - start_time
        print(f"处理完成: {frame_count} 帧, 耗时 {elapsed:.2f}s, 平均 {frame_count/elapsed:.2f} fps")

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description='Jetson Nano 实时异常检测')
    parser.add_argument('--model_path', type=str,
                       default='./results/models/10_gpu0_repeat1/model_s.pth',
                       help='模型文件路径')
    parser.add_argument('--pattern', type=int, default=10,
                       help='异常模式编号')
    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头ID')
    parser.add_argument('--video', type=str, default=None,
                       help='视频文件路径（如果不使用摄像头）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径')
    parser.add_argument('--threshold', type=float, default=None,
                       help='异常检测阈值')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 尝试加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}

    # 创建推理实例
    infer = RealtimeInference(
        model_path=args.model_path,
        pattern=args.pattern,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if args.video:
        # 处理视频文件
        infer.run_video(args.video, args.output)
    else:
        # 实时摄像头推理
        infer.run_camera(camera_id=args.camera, threshold=args.threshold)

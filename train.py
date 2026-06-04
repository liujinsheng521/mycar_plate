"""
YOLOv5m 车牌检测训练脚本 — CCPD 数据集

使用流程:
  1. 准备数据集:  python ccpd_to_yolo.py --ccpd_root /path/to/CCPD --output_dir ./dataset
  2. 开始训练:    python train.py --weights yolov5m.pt --data ccpd.yaml --epochs 100
  3. 导出 ONNX:   python export_onnx.py --weights runs/train/exp/weights/best.pt

注意: 训练需要 GPU，纯 CPU 训练会很慢
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5m on CCPD dataset')
    parser.add_argument('--weights', type=str, default='yolov5m.pt',
                        help='预训练权重路径（默认: yolov5m.pt）')
    parser.add_argument('--data', type=str, default='ccpd.yaml',
                        help='数据集配置文件（默认: ccpd.yaml）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数（默认: 100）')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小（默认: 16）')
    parser.add_argument('--img-size', type=int, default=640,
                        help='训练图像尺寸（默认: 640）')
    parser.add_argument('--yolov5_dir', type=str, default='./yolov5',
                        help='ultralytics/yolov5 源码目录路径')
    parser.add_argument('--device', type=str, default='',
                        help='训练设备，如 cuda:0 或 cpu（默认: 自动选择）')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数（默认: 8）')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='训练输出项目目录（默认: runs/train）')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称（默认: exp）')
    parser.add_argument('--exist-ok', action='store_true',
                        help='允许覆盖已有实验目录，不自增序号')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值，验证集指标连续多少轮不提升则停止（默认: 50）')
    args = parser.parse_args()

    # 将 yolov5 相对路径转为绝对路径
    yolov5_dir = os.path.abspath(args.yolov5_dir)

    # 如果 yolov5 源码不存在，自动 clone
    if not os.path.isdir(yolov5_dir):
        print(f"YOLOv5 directory not found: {yolov5_dir}")
        print("Cloning ultralytics/yolov5...")
        subprocess.run([
            'git', 'clone', 'https://github.com/ultralytics/yolov5.git', yolov5_dir
        ], check=True)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r',
            os.path.join(yolov5_dir, 'requirements.txt')
        ], check=True)

    # 组装 YOLOv5 train.py 调用命令
    cmd = [
        sys.executable, os.path.join(yolov5_dir, 'train.py'),
        '--weights', args.weights,
        '--data', os.path.abspath(args.data),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--imgsz', str(args.img_size),
        '--workers', str(args.workers),
        '--project', os.path.abspath(args.project),
        '--name', args.name,
    ]

    # 根据权重自动选择 YOLOv5m 模型配置
    if args.device:
        cmd.extend(['--device', args.device])
    if args.exist_ok:
        cmd.append('--exist-ok')
    cmd.extend(['--patience', str(args.patience)])

    # 使用 yolov5m 模型结构（由 weights 文件名判断）
    if args.weights and 'yolov5m' in args.weights:
        cmd.extend(['--cfg', os.path.join(yolov5_dir, 'models', 'yolov5m.yaml')])

    print("Running YOLOv5 training:")
    print('  ' + ' '.join(cmd))
    sys.stdout.flush()

    # 执行训练
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

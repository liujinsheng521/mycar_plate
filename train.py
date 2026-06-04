"""
Train YOLOv5m on CCPD license plate dataset.

Usage:
  # Prepare dataset first:
  python ccpd_to_yolo.py --ccpd_root /path/to/CCPD --output_dir ./dataset

  # Train (requires GPU):
  python train.py --weights yolov5m.pt --data ccpd.yaml --epochs 100

  # Export to ONNX after training:
  python export_onnx.py --weights runs/train/exp/weights/best.pt
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5m on CCPD dataset')
    parser.add_argument('--weights', type=str, default='yolov5m.pt',
                        help='Initial weights path (default: yolov5m.pt)')
    parser.add_argument('--data', type=str, default='ccpd.yaml',
                        help='Dataset config path (default: ccpd.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--yolov5_dir', type=str, default='./yolov5',
                        help='Path to ultralytics/yolov5 directory')
    parser.add_argument('--device', type=str, default='',
                        help='Device: cuda:0 or cpu (default: auto)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Data loading workers (default: 8)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory (default: runs/train)')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name (default: exp)')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    args = parser.parse_args()

    # Convert relative yolov5_dir to absolute
    yolov5_dir = os.path.abspath(args.yolov5_dir)

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

    # Build command
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

    # YOLOv5m specific: use the medium model architecture
    # This is handled by the weights file (yolov5m.pt)
    if args.device:
        cmd.extend(['--device', args.device])
    if args.exist_ok:
        cmd.append('--exist-ok')
    cmd.extend(['--patience', str(args.patience)])

    # Update cfg to yolov5m if not using pretrained weights
    if args.weights and 'yolov5m' in args.weights:
        cmd.extend(['--cfg', os.path.join(yolov5_dir, 'models', 'yolov5m.yaml')])

    print("Running YOLOv5 training:")
    print('  ' + ' '.join(cmd))
    sys.stdout.flush()

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

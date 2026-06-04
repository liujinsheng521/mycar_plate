"""
将训练好的 YOLOv5 PyTorch 模型（.pt）导出为 ONNX 格式，方便后续转为 RKNN。

使用方式：
  python export_onnx.py --weights runs/train/exp/weights/best.pt

原理：
  YOLOv5 官方代码中自带了 export.py 导出脚本，这个脚本只是调用它来完成导出。
"""

# 导入需要用到的库
import os                           # 文件路径处理
import sys                          # sys.exit() 退出程序
import argparse                     # 解析命令行参数
import subprocess                   # 在 Python 里执行系统命令（调用别的脚本）


def main():
    # ---- 1. 解析命令行参数 ----
    parser = argparse.ArgumentParser(description='Export YOLOv5 to ONNX')
    # required=True 表示 --weights 是必填的，不传就报错
    parser.add_argument('--weights', type=str, required=True,
                        help='YOLOv5 训练好的模型权重文件路径 (.pt)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图片尺寸（默认 640）')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批次大小（默认 1）')
    parser.add_argument('--yolov5_dir', type=str, default='./yolov5',
                        help='ultralytics/yolov5 项目目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出的 ONNX 文件路径（不传则和 .pt 文件同目录）')
    args = parser.parse_args()

    # ---- 2. 检查文件/目录是否存在 ----
    # os.path.abspath() 把相对路径转成绝对路径，避免路径问题
    yolov5_dir = os.path.abspath(args.yolov5_dir)
    weights_path = os.path.abspath(args.weights)

    # 检查 .pt 权重文件是否存在
    if not os.path.isfile(weights_path):
        print(f"Error: weights not found: {weights_path}")
        sys.exit(1)

    # 检查 YOLOv5 目录是否存在
    if not os.path.isdir(yolov5_dir):
        print(f"Error: YOLOv5 directory not found: {yolov5_dir}")
        sys.exit(1)

    # 检查 YOLOv5 目录里有没有 export.py 这个导出脚本
    export_script = os.path.join(yolov5_dir, 'export.py')
    if not os.path.isfile(export_script):
        print(f"Error: export.py not found in {yolov5_dir}")
        sys.exit(1)

    # ---- 3. 拼接要执行的命令 ----
    # sys.executable 表示当前正在用的 Python 解释器路径
    # 这样能确保调用的是同一个 Python 环境，避免 "找不到模块" 的问题
    cmd = [
        sys.executable, export_script,
        '--weights', weights_path,
        '--include', 'onnx',               # 只导出 ONNX 格式
        '--imgsz', str(args.img_size),      # YOLOv5 export.py 用的是 --imgsz 参数名
        '--batch-size', str(args.batch_size),
        '--opset', '12',                    # opset 12 确保 RKNN 兼容性
    ]

    # ---- 4. 打印命令让用户确认 ----
    print("Exporting to ONNX:")
    print('  ' + ' '.join(cmd))
    sys.stdout.flush()      # 立即把输出刷到终端，防止被缓冲延迟显示

    # ---- 5. 执行命令 ----
    # subprocess.run() 相当于在终端运行这条命令
    # check=True 表示如果命令执行失败（返回非零），自动抛出异常
    subprocess.run(cmd, check=True)

    # ---- 6. 确定输出的 ONNX 文件路径 ----
    # YOLOv5 的 export.py 会把 .onnx 文件生成在 .pt 同目录下
    # 比如 weights/best.pt → weights/best.onnx
    onnx_path = weights_path.replace('.pt', '.onnx')

    # 如果用户指定了 --output，就把生成的 .onnx 复制过去
    if args.output:
        import shutil                      # 文件复制库
        shutil.copy2(onnx_path, args.output)  # copy2 保留文件元信息
        onnx_path = args.output

    print(f"ONNX model exported to: {onnx_path}")


# Python 入口惯例：直接运行时执行 main()，被 import 时不自动执行
if __name__ == '__main__':
    main()

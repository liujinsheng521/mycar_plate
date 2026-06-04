"""
RKNN 模型转换工具：将 YOLOv5 ONNX 模型转换为 RKNN 格式，用于 Rockchip NPU 平台。
"""

# 导入需要用到的库
import os, sys, argparse      # os:文件路径, sys:退出程序, argparse:解析命令行参数
import numpy as np             # numpy:科学计算库,用来创建假数据做验证
from rknn.api import RKNN      # RKNN:瑞芯微提供的核心转换/推理 API

RKNN_MODEL_EXT = '.rknn'       # 最终生成的 RKNN 模型文件后缀
OBJ_THRESH = 0.5               # YOLO 检测置信度阈值（本脚本未实际用到，仅保留参考）
NMS_THRESH = 0.45              # NMS 非极大值抑制阈值（本脚本未实际用到）
IMG_SIZE = 320                 # 默认输入尺寸，验证用。如果模型是 320 请改成 320


def parse_arg():
    """
    解析用户在命令行输入的参数。
    比如用户在终端输入:
      python convert_rknn.py --onnx model/yolov5m.onnx --target rk3588
    这个函数会把 --onnx 和 --target 后面的值提取出来。
    """
    parser = argparse.ArgumentParser(description='Convert YOLOv5 ONNX to RKNN')

    # required=True 表示这个参数必须提供，否则程序报错退出
    parser.add_argument('--onnx', type=str, required=True,
                        help='输入的 ONNX 模型路径')

    # default=None 表示不传这个参数也行
    parser.add_argument('--output', type=str, default=None,
                        help='输出的 RKNN 模型路径（不传则和 ONNX 文件同名）')

    # choices 限制用户只能从列表中选一个目标平台
    parser.add_argument('--target', type=str, default='rk3588',
                        choices=['rk3562', 'rk3566', 'rk3568', 'rk3588',
                                 'rk3588s', 'rv1103', 'rv1106', 'rk2118'],
                        help='目标 NPU 平台（默认 rk3588）')

    # --quant 传了=做量化(模型变小但精度略降),不传=不做量化(FP16精度更高)
    parser.add_argument('--quant', type=str, default=None,
                        help='量化数据集文件路径（如 dataset.txt），不传则不量化')

    # action='store_true' 表示只要写了 --verbose 就是 True，不写就是 False
    parser.add_argument('--verbose', action='store_true',
                        help='是否打印详细的转换日志')

    # 解析参数并返回
    return parser.parse_args()


def main():
    # ---- 1. 解析命令行参数 ----
    args = parse_arg()

    # ---- 2. 检查 ONNX 文件是否存在 ----
    onnx_path = os.path.abspath(args.onnx)          # 把相对路径转成绝对路径
    if not os.path.isfile(onnx_path):               # 判断文件是否存在
        print(f"Error: ONNX model not found: {onnx_path}")
        sys.exit(1)                                 # 不存在就退出，exit(1) 表示出错退出

    # ---- 3. 确定输出 RKNN 文件路径 ----
    if args.output:
        rknn_path = args.output                       # 用户指定了就用自己的
    else:
        # 没指定就用 ONNX 的文件名，但把后缀 .onnx 换成 .rknn
        rknn_path = os.path.splitext(onnx_path)[0] + RKNN_MODEL_EXT

    # ---- 4. 判断是否做量化 ----
    do_quant = args.quant is not None                 # True=量化, False=不量化
    dataset_path = args.quant if do_quant else ''     # 量化需要的数据集文件路径

    # ---- 5. 打印信息让用户确认 ----
    print(f"Converting YOLOv5 to RKNN:")
    print(f"  ONNX:   {onnx_path}")                   # f-string 可以在字符串里直接嵌变量
    print(f"  RKNN:   {rknn_path}")
    print(f"  Target: {args.target}")
    quant_status = 'enabled (int8)' if do_quant else 'disabled (FP16)'
    print(f"  Quant:  {quant_status}")

    # ---- 6. 创建 RKNN 对象 ----
    # verbose=True 会打印详细日志，方便排查问题
    rknn = RKNN(verbose=args.verbose)

    # ==================== 核心转换流程，总共 5 步 ====================

    # ---- 第 1 步：配置模型参数 ----
    # YOLOv5 训练时输入图片被归一化到 [0,1] 范围，方法是：像素值 ÷ 255
    # 所以这里 mean=0, std=255，等价于 (pixel - 0) / 255，把 0~255 映射到 0~1
    print('--> Config model')
    ret = rknn.config(
        mean_values=[[0, 0, 0]],           # RGB 三个通道各减 0
        std_values=[[255, 255, 255]],       # RGB 三个通道各除以 255
        target_platform=args.target,        # 指定部署的 NPU 型号
    )
    # RKNN API 的函数返回 0 表示成功，非 0 表示失败
    if ret != 0:
        print('Config model failed!')
        sys.exit(ret)                       # 失败就退出，退出码是错误码
    print('done')

    # ---- 第 2 步：加载 ONNX 模型 ----
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX failed!')
        sys.exit(ret)
    print('done')

    # ---- 第 3 步：构建（转换）RKNN 模型 ----
    # 这是最核心的一步，RKNN-Toolkit 会把 ONNX 模型解析、优化并转成 NPU 能运行的格式
    # do_quantization=True 时会对权重做 int8 量化，模型更小、速度更快，但精度会稍微降低
    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_path)
    if ret != 0:
        print('Build RKNN failed!')
        sys.exit(ret)
    print('done')

    # ---- 第 4 步：导出 .rknn 文件 ----
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_path)       # 把转换好的模型保存到磁盘
    if ret != 0:
        print('Export RKNN failed!')
        sys.exit(ret)
    print('done')

    # ---- 第 5 步：用假数据做一次推理验证 ----
    # 确认转换后的模型能正常跑，输出形状也是预期的
    print('--> Verify model with dummy input')
    # 初始化运行时（在 PC 端模拟 NPU 运行，如果当前电脑没有 NPU 驱动也能以模拟模式运行）
    ret = rknn.init_runtime()
    if ret != 0:
        # 纯 PC 环境没有 NPU 硬件，初始化失败是正常的，不影响模型转换
        print('Init runtime failed (this is ok if run on PC without NPU)')
    else:
        # 创建全黑图片做推理验证，尺寸需匹配模型输入
        try:
            dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            outputs = rknn.inference(inputs=[dummy], data_format=['nhwc'])
        except ValueError as e:
            if 'shape' in str(e):
                # 模型实际尺寸与 IMG_SIZE 不一致，自动修正
                msg = str(e)
                import re
                # 找 "expect nhwc like (1, H, W, 3)"
                match = re.search(r"expect.*?\(1, (\d+), (\d+), 3\)", msg)
                if match:
                    h, w = int(match.group(1)), int(match.group(2))
                    dummy = np.zeros((1, h, w, 3), dtype=np.uint8)
                    outputs = rknn.inference(inputs=[dummy], data_format=['nhwc'])
                else:
                    raise e
            else:
                raise e
        # 打印输出形状，比如 (1, 25200, 85) 表示检测到 25200 个候选框
        print(f'  Output shape: {outputs[0].shape}')
        print(f'  Output dtype: {outputs[0].dtype}')
        print('Verify ok')
        rknn.release()                      # 释放资源

    print(f"\nRKNN model saved to: {rknn_path}")
    print(f"\nTo deploy on board, copy {rknn_path} to the board and use C++ API.")


# 这是 Python 程序的入口惯例：
# 当直接运行 python convert_rknn.py 时，__name__ 等于 '__main__'，会执行 main()
# 当被其他文件 import 时，__name__ 不等于 '__main__'，不会自动执行 main()
if __name__ == '__main__':
    main()

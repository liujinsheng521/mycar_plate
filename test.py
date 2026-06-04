"""
YOLOv5 车牌检测 — RKNN 推理演示

支持两种模式：
  1. PC 模拟模式：用 --onnx 指定 ONNX 模型，rknn.build 后在本机仿真运行
  2. 板端部署模式：用 --model 指定已编译的 .rknn，连接 NPU 板子推理

用法:
  python test.py --model model.rknn --image test.jpg
  python test.py --onnx model.onnx --image test.jpg --target rk3588
"""

import os
import sys
import argparse
import cv2
import numpy as np
from rknn.api import RKNN

OBJ_THRESH = 0.5    # 检测置信度阈值
NMS_THRESH = 0.45   # NMS IoU 阈值
IMG_SIZE = 320      # 模型输入尺寸


def letterbox(im, new_shape=(320, 320), color=(114, 114, 114)):
    """
    等比例缩放 + 灰边填充（Letterbox）
    保持原图宽高比，不足部分用 color 填充，返回填充后的图像及缩放参数

    返回: (padded_img, ratio, (dw, dh))
      - ratio: 缩放比例
      - dw, dh: 水平/垂直方向的填充像素数
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def nms_boxes(boxes, scores):
    """
    非极大值抑制（NMS）
    按置信度降序排序，逐个保留高分框，抑制与其 IoU 超过阈值的重叠框
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def post_process(outputs, img_shape, ratio, pad):
    """
    YOLOv5 输出后处理

    输入输出格式:
      - outputs[0]: [1, N, 6]，每行 [cx, cy, w, h, conf, cls]
      - ratio: letterbox 缩放比例
      - pad: letterbox 填充 (dw, dh)

    处理步骤:
      1. 置信度过滤（OBJ_THRESH）
      2. 中心坐标 → 左上右下坐标
      3. 反算 letterbox，还原到原图坐标系
      4. 坐标裁剪到原图边界
      5. NMS 去重
    """
    predictions = outputs[0][0]
    mask = predictions[:, 4] >= OBJ_THRESH
    predictions = predictions[mask]
    if len(predictions) == 0:
        return None, None

    boxes = predictions[:, :4]
    scores = predictions[:, 4]

    # 中心坐标 (cx,cy,w,h) → 左上右下 (x1,y1,x2,y2)
    xyxy = np.copy(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # 去除 letterbox padding，缩放回原图
    dw, dh = pad
    xyxy[:, 0] = (xyxy[:, 0] - dw) / ratio
    xyxy[:, 2] = (xyxy[:, 2] - dw) / ratio
    xyxy[:, 1] = (xyxy[:, 1] - dh) / ratio
    xyxy[:, 3] = (xyxy[:, 3] - dh) / ratio

    # 裁剪到原图边界
    h_img, w_img = img_shape[:2]
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w_img)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, w_img)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h_img)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, h_img)

    # NMS
    keep = nms_boxes(xyxy, scores)
    if len(keep) == 0:
        return None, None

    return xyxy[keep], scores[keep]


def draw_results(img, boxes, scores):
    """
    在图像上绘制检测框及置信度标签
    同时将每个车牌区域裁剪保存为单独的文件
    """
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"plate {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [int(v) for v in box]
        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size > 0:
            crop_path = f"plate_{i}_conf_{score:.3f}.jpg"
            cv2.imwrite(crop_path, plate_crop)
            print(f"  Cropped plate saved: {crop_path}")

    return img


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 License Plate Detection with RKNN')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to RKNN model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--save', type=str, default='result.jpg',
                        help='Output image path (default: result.jpg)')
    parser.add_argument('--target', type=str, default=None,
                        help='Target platform (e.g. rk3588). Run on PC if not set.')
    parser.add_argument('--onnx', type=str, default=None,
                        help='ONNX model path (for PC simulator, use with --target not set)')
    args = parser.parse_args()

    if not args.model and not args.onnx:
        print('Error: Either --model or --onnx must be provided.')
        sys.exit(1)

    rknn = RKNN(verbose=False)

    if args.onnx:
        # PC 模拟模式：从 ONNX 开始，rknn.load_onnx + rknn.build，无需物理板子
        print(f'Loading ONNX: {args.onnx}')
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                    target_platform=args.target or 'rk3588')
        rknn.load_onnx(model=args.onnx)
        rknn.build(do_quantization=False)
        ret = rknn.init_runtime()
    else:
        # 板端模式：加载已编译的 .rknn 模型，需要连接 NPU 板或使用模拟器
        print(f'Loading RKNN model: {args.model}')
        ret = rknn.load_rknn(args.model)
        if ret != 0:
            print('Load RKNN failed!')
            sys.exit(1)
        if args.target:
            ret = rknn.init_runtime(target=args.target)
        else:
            ret = rknn.init_runtime()

    if ret != 0:
        print('Init runtime failed!')
        print('Tip: For PC simulator, use --onnx instead of --model')
        sys.exit(1)
    print('RKNN model loaded successfully')

    # 读取输入图片
    img_src = cv2.imread(args.image)
    if img_src is None:
        print(f'Failed to read image: {args.image}')
        sys.exit(1)
    print(f'Input image: {args.image} ({img_src.shape[1]}x{img_src.shape[0]})')

    # 预处理：BGR → RGB → Letterbox → NHWC
    img_rgb = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_padded, ratio, pad = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_padded, 0).astype(np.uint8)

    # NPU 推理
    print('Running inference...')
    outputs = rknn.inference(inputs=[img_input], data_format=['nhwc'])
    print(f'Output shape: {outputs[0].shape}')

    # 后处理：解码 + NMS
    boxes, scores = post_process(outputs, img_src.shape, ratio, pad)
    if boxes is None:
        print('No license plates detected.')
    else:
        print(f'Detected {len(boxes)} license plate(s):')
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = [int(v) for v in box]
            print(f'  [{i}] plate (conf: {score:.3f}) box: ({x1},{y1},{x2},{y2})')

        # 绘制并保存结果图
        result_img = draw_results(img_src.copy(), boxes, scores)
        cv2.imwrite(args.save, result_img)
        print(f'\nResult saved to {args.save}')

    # 单独裁剪保存每个车牌区域
    if boxes is not None:
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = [int(v) for v in box]
            plate_crop = img_src[y1:y2, x1:x2]
            if plate_crop.size > 0:
                crop_path = f"plate_{i}.jpg"
                cv2.imwrite(crop_path, plate_crop)
                print(f"  Extracted plate saved: {crop_path}")

    rknn.release()


if __name__ == '__main__':
    main()

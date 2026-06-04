# mycar_plate — 车牌检测与识别 (RKNN)

基于 YOLOv5 + LPRNet 的车牌检测和字符识别项目，部署在 Rockchip RK3588 NPU 平台。

## 整体流程

```
CCPD 数据集
    │
    ▼
ccpd_to_yolo.py ─── 将 CCPD 转为 YOLOv5 训练格式
    │
    ▼
train.py ─────────────── 训练 YOLOv5 车牌检测模型
    │                        (输出 .pt 权重文件)
    ▼
export_onnx.py ───────── 导出 ONNX 格式
    │                        (调用 yolov5/export.py)
    ▼
convert_rknn.py ──────── 转为 RKNN 格式
    │                        可选 int8 量化
    ▼
test.py ──────────────── PC 端 Python 推理验证
    │
    ▼
cpp/build-linux.sh ───── 交叉编译 C++ 部署程序
    │
    ▼
开发板上运行 ──────────── rknn_plate_detection_demo
```

---

## 1. 环境准备

### 1.1 激活环境

```bash
conda activate RKNN-Toolkit2
```

### 1.2 项目目录结构

```
mycar_plate/
├── ccpd_to_yolo.py        # CCPD → YOLOv5 格式转换
├── ccpd.yaml              # 数据集配置
├── train.py               # YOLOv5 训练脚本
├── export_onnx.py         # PyTorch → ONNX 导出
├── convert_rknn.py        # ONNX → RKNN 转换
├── test.py                # PC 端 Python 推理
├── dataset.txt            # 量化校准图片列表
├── dataset/               # 数据集目录
│   └── images/
│       ├── train/         # 训练图片
│       └── val/           # 验证图片
├── model/                 # 模型文件
│   ├── yolov5n_plate.rknn # 原版好使的 RKNN 模型
│   ├── best.rknn          # 新生成的 RKNN 模型
│   ├── best.onnx          # ONNX 模型
│   └── ...
├── runs/train/            # 训练产出
│   └── plate_30epoch/
│       ├── weights/
│       │   ├── best.pt    # 训练好的权重
│       │   └── last.pt    # 最后一个 epoch 的权重
│       ├── results.csv    # 训练指标
│       ├── opt.yaml       # 训练参数配置
│       └── ...
├── test_data/             # 测试图片
│   ├── IMG.jpg
│   └── 皖A09N61.jpg
├── yolov5/                # ultralytics/yolov5 子项目
├── cpp/                   # C++ 部署代码
│   ├── build-linux.sh     # 交叉编译脚本
│   ├── CMakeLists.txt     # CMake 构建配置
│   ├── main.cc            # 主程序入口
│   ├── plate_detector.cc  # 车牌检测器
│   ├── plate_detector.h
│   ├── lprnet.cc          # 车牌字符识别
│   └── lprnet.h
└── my_lprnet/             # LPRNet 训练代码
    └── LPRNet_Pytorch/
```

---

## 2. 数据集准备

### 2.1 CCPD → YOLOv5 格式转换

```bash
python ccpd_to_yolo.py \
    --ccpd_root /path/to/CCPD \
    --output_dir ./dataset
```

- 从 CCPD 文件名中解析车牌框坐标
- 转为 YOLOv5 归一化格式：`<class> <cx_norm> <cy_norm> <w_norm> <h_norm>`
- 按 9:1 分割训练/验证集
- 生成 `dataset/dataset.yaml`

### 2.2 量化校准图片

`dataset.txt` 中填写图片路径（每行一张），用于 int8 量化校准：

```
./dataset/images/train/xxx.jpg
./dataset/images/train/xxx.jpg
...
```

> **注意**：`dataset.txt` 中不能有 `#` 注释行，RKNN 无法解析。

---

## 3. 模型训练

### 3.1 训练命令

```bash
python train.py \
    --weights yolov5n.pt \
    --data ccpd.yaml \
    --epochs 100 \
    --batch-size 8 \
    --img-size 320 \
    --device cuda:0
```

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `yolov5n.pt` | 预训练权重，可选 `yolov5s.pt` / `yolov5m.pt` |
| `--data` | `ccpd.yaml` | 数据集配置文件 |
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--img-size` | 640 | **训练图片尺寸，板端部署必须一致** |
| `--device` | 自动 | `cuda:0` 或 `cpu` |

### 3.3 训练输出

每个 epoch 在 `runs/train/exp/` 下输出指标和权重文件：

- `weights/best.pt` — 验证集上最优的权重
- `weights/last.pt` — 最后一个 epoch 的权重
- `results.csv` — 完整训练日志
- `results.png` — 指标曲线图

### 3.4 当前训练记录

项目自带一个 30 epoch 的训练结果（YOLOv5n 模型）：

```
runs/train/plate_30epoch/
├── weights/best.pt    # 3.7 MB
└── opt.yaml           # 训练参数：imgsz=320, epochs=30, batch=8
```

最终指标：**mAP@0.5 = 0.994**, **Precision = 0.989**, **Recall = 1.0**

> **重要**：训练时的 `--img-size` 参数决定了后续导出和转换的输入尺寸，必须保持一致。

---

## 4. 导出 ONNX

### 4.1 导出命令

```bash
python export_onnx.py \
    --weights runs/train/plate_30epoch/weights/best.pt \
    --img-size 320          # 必须与训练时的 imgsz 一致！
```

### 4.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | 必填 | 训练好的 `.pt` 权重路径 |
| `--img-size` | 640 | **必须与训练 `--img-size` 一致** |
| `--batch-size` | 1 | 批次大小 |
| `--yolov5_dir` | `./yolov5` | YOLOv5 项目目录 |

### 4.3 原理

`export_onnx.py` 调用 `yolov5/export.py` 完成导出，固定使用 **opset 12**（高于 12 的 opset 转 RKNN 后在 NPU 上可能出现兼容性问题）。导出后的 `.onnx` 生成在与 `.pt` 同目录。

> **常见错误**：导出时 `--img-size` 必须与训练时一致。训练用 320，导出用 640，生成的 RKNN 在 NPU 上会检测异常。

---

## 5. 转换 RKNN

### 5.1 基本转换（FP16，推荐）

```bash
python convert_rknn.py \
    --onnx runs/train/plate_30epoch/weights/best.onnx \
    --target rk3588

# 拷贝到 model/ 目录，方便后续部署使用
cp runs/train/plate_30epoch/weights/best.rknn model/
```

### 5.2 量化转换（int8，精度略降）

```bash
python convert_rknn.py \
    --onnx runs/train/plate_30epoch/weights/best.onnx \
    --target rk3588 \
    --quant dataset.txt
```

### 5.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--onnx` | 必填 | 输入的 ONNX 模型路径 |
| `--target` | `rk3588` | NPU 平台（rk3566/rk3588/rv1106 等） |
| `--output` | 自动 | 输出 RKNN 路径，不传则与 ONNX 同文件 |
| `--quant` | 不量化 | 量化校准图片列表，加上则做 int8 量化 |
| `--verbose` | 关闭 | 打印详细转换日志 |

### 5.4 转换输出

转换成功后在控制台可看到验证信息：

```
Output shape: (1, 6300, 6)    # 320×320 输入的输出形状
Output dtype: float32
Verify ok
RKNN model saved to: .../best.rknn
```

不同输入尺寸的输出形状：

| 输入尺寸 | 输出形状 | 描述 |
|----------|----------|------|
| 320×320 | (1, 6300, 6) | 40×40 + 20×20 + 10×10=6300 个预测框 |
| 640×640 | (1, 25200, 6) | 80×80 + 40×40 + 20×20=25200 个预测框 |

输出格式 `[cx, cy, w, h, conf, cls]`：
- `cx, cy` — 归一化中心坐标
- `w, h` — 归一化宽高
- `conf` — 检测置信度
- `cls` — 类别（车牌=0）

### 5.5 FP16 vs int8 选择

| 类型 | 大小 | 精度 | 速度 | 推荐场景 |
|------|------|------|------|----------|
| **FP16**（不量化） | 4.6 MB | 高 | 快 | **默认推荐** |
| **int8**（量化） | 3.1 MB | 略降 | 更快 | 存储/带宽受限时 |

YOLOv5n 本身已属于轻量模型，FP16 即可满足大多数场景。

---

## 6. PC 端 Python 推理验证

### 6.1 PC 模拟模式（无需接板子）

```bash
python test.py \
    --onnx runs/train/plate_30epoch/weights/best.onnx \
    --image test_data/IMG.jpg \
    --save result.jpg
```

### 6.2 板端推理模式（需通过 ADB 连接开发板）

```bash
python test.py \
    --model model/best.rknn \
    --image test_data/IMG.jpg \
    --target rk3588 \
    --save result.jpg
```

### 6.3 参数说明

| 参数 | 说明 |
|------|------|
| `--model` | RKNN 模型路径（板端模式） |
| `--onnx` | ONNX 模型路径（PC 模拟模式，无需接板子） |
| `--image` | 输入图片路径 |
| `--save` | 结果保存路径，默认 `result.jpg` |
| `--target` | 指定 NPU 平台如 `rk3588`（不传则 PC 模拟） |

> PC 模拟模式不支持 `load_rknn` 加载板端 RKNN 模型，必须用 `--onnx` 方式。

---

## 7. C++ 交叉编译与板端部署

### 7.1 交叉编译

```bash
cd cpp
./build-linux.sh rk3588
```

脚本执行流程：

1. 使用 SDK 提供的交叉编译器 `aarch64-none-linux-gnu-gcc`
2. CMake 构建
3. 编译出 `rknn_plate_detection_demo` 可执行文件
4. 自动拷贝 `model/*.rknn` 和 `test_data/` 到输出目录
5. 拷贝运行时库 `librknnrt.so`、`librga.so`

编译产物在 `build/build_rk3588_linux_aarch64_Release/install/rknn_plate_detection_demo/`。

### 7.2 前提条件

- 交叉编译器：`prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu`
- 依赖：`rknn_model_zoo-2.0.0` 中的 3rdparty 库（rknpu2、librga、OpenCV）

### 7.3 在开发板上运行

```bash
# 将编译产物拷贝到开发板
scp -r build/build_rk3588_linux_aarch64_Release/install/rknn_plate_detection_demo/ root@板子IP:/data/

# 登录板子运行
ssh root@板子IP
cd /data/rknn_plate_detection_demo

# 车牌检测 + 识别（需要两个模型）
./rknn_plate_detection_demo model/yolov5n_plate.rknn model/lprnet.rknn model/IMG.jpg
```

### 7.4 程序流程

```
main.cc:
1. init_plate_detector()  → 加载检测 RKNN 模型
2. init_lprnet()          → 加载识别 RKNN 模型
3. read_image()           → 读取输入图片
4. inference_plate_detector() → YOLOv5 检测
5. 对每个检测到的车牌框:
   → 裁剪车牌区域
   → inference_lprnet()   → LPRNet 识别字符
   → 打印车牌号
6. 释放资源
```

---

## 8. 重要注意事项

### 8.1 输入尺寸必须一致

训练、导出 ONNX、转换 RKNN 三个环节的输入尺寸必须一致：

```
train.py    --img-size 320
export_onnx.py --img-size 320
convert_rknn.py  ← 自动读取 ONNX 输入尺寸
```

这是最常见的问题——训练用 320 但导出用 640，会导致 NPU 上检测效果异常。

### 8.2 量化精度损失

YOLOv5n 参数量仅 176 万，int8 量化的精度损失可能较明显。原版 `yolov5n_plate.rknn` 使用 FP16 未量化。建议先试用 FP16，仅在存储/性能不足时尝试量化。

### 8.3 dataset.txt 格式

量化校准文件只包含图片路径，不能有注释行：

```
✔ ./dataset/images/train/xxx.jpg
✘ # 这是注释（会导致 Unsupport file #! 错误）
```

### 8.4 模型文件参考

| 文件 | 输入尺寸 | 量化 | 大小 | 备注 |
|------|----------|------|------|------|
| `yolov5n_plate.rknn` | 320×320 | FP16 | 4.6 MB | 已验证好使 |
| `best.rknn` | 320×320 | FP16 | 4.6 MB | 新生成 |
| `test_int8.rknn` | 320×320 | int8 | 3.1 MB | 量化版 |
| `anpr_yolov5s_fixed.rknn` | 640×640 | — | 15.9 MB | YOLOv5s 大模型 |

---

## 9. 完整命令速查

```bash
# 0. 激活环境
conda activate RKNN-Toolkit2

# 1. 数据转换
python ccpd_to_yolo.py --ccpd_root /path/CCPD --output_dir ./dataset

# 2. 训练
python train.py --weights yolov5n.pt --data ccpd.yaml --epochs 100 --img-size 320

# 3. 导出 ONNX
python export_onnx.py --weights runs/train/exp/weights/best.pt --img-size 320

# 4. 转 RKNN（FP16，推荐）
python convert_rknn.py --onnx runs/train/exp/weights/best.onnx --target rk3588

# 4b. 拷贝到 model/ 目录
cp runs/train/exp/weights/best.rknn model/

# 4c. 转 RKNN（int8 量化）
python convert_rknn.py --onnx runs/train/exp/weights/best.onnx --target rk3588 --quant dataset.txt

# 4d. 拷贝 int8 模型
cp runs/train/exp/weights/best.rknn model/best_int8.rknn

# 5. Python 验证（PC 模拟，无需板子）
python test.py --onnx model/best.onnx --image test_data/test.jpg

# 5b. Python 验证（接板子）
python test.py --model model/best.rknn --image test_data/test.jpg --target rk3588    #板端需部署toolkit

# 6. C++ 编译
cd cpp && ./build-linux.sh rk3588

# 7. 板端推理运行
./rknn_plate_detection_demo model/best.rknn model/lprnet.rknn model/test.jpg

```

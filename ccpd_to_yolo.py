"""
将 CCPD 车牌数据集转换为 YOLOv5 训练格式。

CCPD 数据集的文件名里编码了标注信息（车牌位置、车牌号等），格式如下：
  [区域]_[倾斜角]_[车牌框]_[关键点]_[宽高比]_[亮度]_[模糊]_[车牌字符].jpg
本脚本只提取其中的车牌框（第 3 段），转成 YOLOv5 的归一化坐标格式。

YOLOv5 标签格式（每张图对应一个 .txt 文件）：
  <类别id> <中心点x_norm> <中心点y_norm> <宽_norm> <高_norm>
本脚本中类别固定为 0，代表 "license_plate"。
"""

# 导入需要用到的库
import os                           # 文件路径拼接、目录遍历
import sys                          # 系统功能（本脚本主要用于 exit）
import shutil                       # 复制文件（shutil.copy2 保留文件元信息）
import random                       # 打乱数据集顺序，随机分割训练/验证集
import argparse                     # 解析用户在命令行输入的参数
from glob import glob               # 用通配符批量查找文件，比如找所有 .jpg


def parse_ccpd_filename(filename):
    """
    从 CCPD 文件名中解析出车牌框的坐标。

    CCPD 文件名示例：
      025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
      第 3 段（用 - 分隔）就是车牌框：154&383_386&473
      表示左上角 (154, 383)，右下角 (386, 473)

    参数：
        filename: 图片文件名（如 "xxx.jpg"）

    返回：
        如果解析成功 -> (x1, y1, x2, y2) 四个整数坐标
        如果解析失败 -> None
    """
    # 去掉文件后缀，只保留名字部分
    basename = os.path.splitext(filename)[0]
    # 用 "-" 分割文件名各字段
    fields = basename.split('-')
    # 至少有 4 段才包含有效的车牌框信息
    if len(fields) < 4:
        return None

    # 第 3 段（下标 2）就是 bbox，格式如 "154&383_386&473"
    bbox_str = fields[2]
    # 用 "_" 分割左上角和右下角
    parts = bbox_str.split('_')
    if len(parts) != 2:
        return None

    # 左上角 "154&383" -> x1=154, y1=383
    x1_str, y1_str = parts[0].split('&')
    # 右下角 "386&473" -> x2=386, y2=473
    x2_str, y2_str = parts[1].split('&')

    return int(x1_str), int(y1_str), int(x2_str), int(y2_str)


def convert_ccpd_to_yolo(ccpd_root, output_dir, val_ratio=0.1, subsets=None):
    """
    将 CCPD 数据集转换为 YOLOv5 训练格式。

    主要工作：
      1. 遍历 CCPD 数据集的所有图片
      2. 从文件名里解析出车牌框坐标
      3. 读取图片获取真实宽高
      4. 将坐标转为 YOLOv5 的归一化格式
      5. 将图片和标签文件分别复制到 images/ 和 labels/ 目录
      6. 按比例分割训练集和验证集
      7. 生成 dataset.yaml 配置文件

    参数：
        ccpd_root: CCPD 数据集根目录路径
        output_dir: YOLOv5 格式输出目录路径
        val_ratio: 验证集比例，默认 0.1（即 10% 的数据做验证）
        subsets: 要处理的子集列表，默认自动检测所有子目录
    """
    # ---- 1. 确定要处理的子集 ----
    if subsets is None:
        # 自动检测 ccpd_root 下一级的所有子目录作为子集
        # 比如 ccpd_base、ccpd_weather、ccpd_night 等
        subsets = [d for d in os.listdir(ccpd_root)
                   if os.path.isdir(os.path.join(ccpd_root, d))]
        if not subsets:
            # 如果没有子目录，说明图片直接在根目录下
            subsets = ['']

    # ---- 2. 创建输出目录结构 ----
    # YOLOv5 要求的目录格式：
    #   output/
    #     images/train/    -> 训练图片
    #     images/val/      -> 验证图片
    #     labels/train/    -> 训练标签
    #     labels/val/      -> 验证标签
    img_dir = os.path.join(output_dir, 'images', 'train')
    lbl_dir = os.path.join(output_dir, 'labels', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    val_lbl_dir = os.path.join(output_dir, 'labels', 'val')
    for d in [img_dir, lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)      # exist_ok=True 表示目录已存在也不报错

    total_images = 0    # 总共找到多少张图片
    converted = 0       # 成功转换了多少张
    skipped = 0         # 跳过了多少张（解析失败或图片损坏）

    # ---- 3. 逐个处理每个子集 ----
    for subset in subsets:
        # 拼接子集路径
        if subset:
            subset_path = os.path.join(ccpd_root, subset)
        else:
            subset_path = ccpd_root

        # 如果子集路径不存在，跳过
        if not os.path.isdir(subset_path):
            print(f"  Subset not found: {subset_path}, skipping")
            continue

        # 用 glob 查找该子集下所有 .jpg 文件
        # sorted 排序保证每次运行顺序一致（虽然后面会用 random.shuffle 打乱）
        jpg_files = sorted(glob(os.path.join(subset_path, '*.jpg')))
        print(f"  Subset '{subset or '(root)'}': {len(jpg_files)} images found")
        total_images += len(jpg_files)

        # ---- 4. 分割训练集 / 验证集 ----
        # 先把所有图片顺序打乱，避免数据集本身有顺序偏差
        random.shuffle(jpg_files)
        # 计算验证集数量。比如 10000 张图 * 0.1 = 1000 张做验证
        n_val = int(len(jpg_files) * val_ratio)

        # ---- 5. 逐张处理图片 ----
        for i, img_path in enumerate(jpg_files):
            fname = os.path.basename(img_path)       # 只取文件名，不要路径

            # 从文件名解析车牌框坐标
            bbox = parse_ccpd_filename(fname)
            if bbox is None:
                skipped += 1
                continue          # 解析失败就跳过

            x1, y1, x2, y2 = bbox

            # 使用 cv2 读取图片，获取真实宽高
            # 注意：import 写在函数内部，表示用到这里才导入 cv2
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue          # 图片损坏或无法读取，跳过

            # img.shape 返回值：(高, 宽, 通道数)，这里只取高和宽
            h_img, w_img = img.shape[:2]

            # 边界保护：有时 CCPD 的标注框会稍微超出图片边缘
            # 把坐标限制在图片范围内，防止训练时出错
            x1 = max(0, x1)               # 左边界不能小于 0
            y1 = max(0, y1)               # 上边界不能小于 0
            x2 = min(w_img, x2)           # 右边界不能超过图片宽度
            y2 = min(h_img, y2)           # 下边界不能超过图片高度

            # 如果裁剪后框的宽或高 <= 0，说明标注无效，跳过
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            # ---- 6. 转换为 YOLOv5 格式 ----
            # YOLOv5 用归一化的中心点坐标 + 归一化宽高，而不是直接用像素坐标
            # 这样不管图片尺寸是多少，标注值都在 0~1 之间
            box_w = x2 - x1               # 车牌框的像素宽度
            box_h = y2 - y1               # 车牌框的像素高度
            cx = (x1 + x2) / 2.0 / w_img  # 中心点 x（归一化到 0~1）
            cy = (y1 + y2) / 2.0 / h_img  # 中心点 y（归一化到 0~1）
            nw = box_w / w_img            # 宽度（归一化到 0~1）
            nh = box_h / h_img            # 高度（归一化到 0~1）

            # 拼成一行标签文本，class_id 固定为 0（只有"车牌"这一个类别）
            # 保留 6 位小数够用，又不至于文件太大
            label_line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"

            # ---- 7. 决定放到训练集还是验证集 ----
            # 前 n_val 张（已经随机打乱了）放验证集，其余放训练集
            if i < n_val:
                target_img_dir = val_img_dir
                target_lbl_dir = val_lbl_dir
            else:
                target_img_dir = img_dir
                target_lbl_dir = lbl_dir

            # 复制图片到目标目录
            # shutil.copy2 和 shutil.copy 类似，但会保留文件的修改时间等元信息
            shutil.copy2(img_path, os.path.join(target_img_dir, fname))

            # 写标签文件：图片 xxx.jpg 对应标签 xxx.txt
            label_path = os.path.join(target_lbl_dir, os.path.splitext(fname)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(label_line)

            converted += 1

            # 每处理 5000 张打印一次进度，让用户知道还在跑
            if converted % 5000 == 0:
                print(f"  Processed {converted} images...")

    # ---- 8. 打印统计信息 ----
    print(f"\nDone! Total: {total_images}, Converted: {converted}, Skipped: {skipped}")
    n_train = len(os.listdir(img_dir))       # 统计训练集图片数量
    n_val = len(os.listdir(val_img_dir))     # 统计验证集图片数量
    print(f"  Train: {n_train} images, Val: {n_val} images")

    # ---- 9. 生成 dataset.yaml 配置文件 ----
    # YOLOv5 训练时需要这个文件告诉它数据在哪里、有多少个类别
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"# CCPD dataset for YOLOv5\n")
        f.write(f"train: {os.path.abspath(output_dir)}/images/train\n")
        f.write(f"val: {os.path.abspath(output_dir)}/images/val\n")
        f.write(f"\nnc: 1\n")                                    # num_classes = 1（只有车牌）
        f.write(f"names: ['license_plate']\n")                   # 类别名字列表
    print(f"  Dataset config: {yaml_path}")


# 程序入口
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Convert CCPD dataset to YOLOv5 format')
    parser.add_argument('--ccpd_root', type=str, required=True,
                        help='CCPD 数据集根目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='YOLOv5 格式输出目录路径')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例（默认 0.1，即 10%% 做验证）')
    parser.add_argument('--subsets', type=str, nargs='*', default=None,
                        help='要处理的子集列表（如 ccpd_base ccpd_weather），'
                             '不传则自动检测所有子目录')
    args = parser.parse_args()

    # 固定随机种子，保证每次运行分割结果一致
    random.seed(42)

    # 开始转换
    convert_ccpd_to_yolo(args.ccpd_root, args.output_dir, args.val_ratio, args.subsets)

/*
 * 车牌检测 + 识别演示程序
 * 用法: ./rknn_plate_detection_demo <detect.rknn> <lprnet.rknn> <image.jpg>
 *
 * 处理流程：
 *   1. 分别初始化 YOLOv5 检测器 和 LPRNet 识别器
 *   2. 读取输入图片
 *   3. YOLOv5 推理获取车牌边界框
 *   4. 对每个检测框裁剪车牌区域
 *   5. LPRNet 推理识别车牌字符
 *   6. 打印结果
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plate_detector.h"
#include "lprnet.h"
#include "file_utils.h"
#include "image_utils.h"

int main(int argc, char **argv)
{
    /* 检查命令行参数：检测模型 + 识别模型 + 图片路径 */
    if (argc != 4)
    {
        printf("Usage: %s <detect_model.rknn> <lprnet_model.rknn> <image.jpg>\n", argv[0]);
        printf("Example:\n");
        printf("  %s model/anpr_yolov5s_fixed.rknn model/lprnet.rknn model/IMG.jpg\n", argv[0]);
        return -1;
    }

    const char *det_model_path = argv[1];
    const char *lpr_model_path = argv[2];
    const char *image_path = argv[3];

    /* ===== 1. 初始化 YOLOv5 车牌检测器 ===== */
    printf(">>> Init plate detector: %s\n", det_model_path);
    plate_detector_context_t det_ctx;
    memset(&det_ctx, 0, sizeof(det_ctx));
    int ret = init_plate_detector(det_model_path, &det_ctx);
    if (ret != 0) return -1;

    /* ===== 2. 初始化 LPRNet 车牌识别器 ===== */
    printf(">>> Init LPRNet: %s\n", lpr_model_path);
    lprnet_context_t lpr_ctx;
    memset(&lpr_ctx, 0, sizeof(lpr_ctx));
    ret = init_lprnet(lpr_model_path, &lpr_ctx);
    if (ret != 0) { release_plate_detector(&det_ctx); return -1; }

    /* ===== 3. 读取输入图片 ===== */
    printf(">>> Read image: %s\n", image_path);
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0) { printf("read_image fail!\n"); goto out; }
    printf("  image size: %dx%d\n", src_image.width, src_image.height);

    /* ===== 4. YOLOv5 推理：检测车牌位置 ===== */
    printf(">>> Running detection...\n");
    plate_det_results_t det_results;
    ret = inference_plate_detector(&det_ctx, &src_image, &det_results);
    if (ret != 0) { printf("detect fail!\n"); goto out; }

    if (det_results.count == 0)
    {
        printf("  No plates detected.\n");
        goto out;
    }

    printf("  Detected %d candidate(s):\n", det_results.count);

    /* ===== 5. 遍历每个检测框 → 裁剪 → LPRNet 识别 ===== */
    for (int i = 0; i < det_results.count; i++)
    {
        plate_det_result_t &det = det_results.results[i];

        /* 裁剪边界到原图范围内，过滤无效框 */
        int x1 = det.left < 0 ? 0 : det.left;
        int y1 = det.top < 0 ? 0 : det.top;
        int x2 = det.right > src_image.width ? src_image.width : det.right;
        int y2 = det.bottom > src_image.height ? src_image.height : det.bottom;
        if (x2 <= x1 || y2 <= y1) continue;

        int crop_w = x2 - x1;
        int crop_h = y2 - y1;
        /* 过滤过小检测框（面积 < 原图 0.05%） */
        if (crop_w * crop_h < src_image.width * src_image.height * 0.0005) continue;

        /* 从原图逐行拷贝车牌区域到连续缓冲区 */
        unsigned char *crop_buf = (unsigned char *)malloc(crop_w * crop_h * 3);
        if (!crop_buf) continue;
        unsigned char *src_row = src_image.virt_addr + (y1 * src_image.width + x1) * 3;
        for (int r = 0; r < crop_h; r++) {
            memcpy(crop_buf + r * crop_w * 3, src_row + r * src_image.width * 3, crop_w * 3);
        }

        /* 构造 LPRNet 输入图像 */
        image_buffer_t plate_img;
        memset(&plate_img, 0, sizeof(plate_img));
        plate_img.width = crop_w;
        plate_img.height = crop_h;
        plate_img.format = IMAGE_FORMAT_RGB888;
        plate_img.size = crop_w * crop_h * 3;
        plate_img.virt_addr = crop_buf;

        /* LPRNet 推理 */
        lprnet_result_t lpr_result;
        ret = inference_lprnet(&lpr_ctx, &plate_img, &lpr_result);
        if (ret != 0) { free(crop_buf); continue; }

        printf("  [%d] Plate: %s (conf: %.3f)  box: (%d,%d,%d,%d)\n",
               i, lpr_result.plate_name.c_str(), det.confidence,
               x1, y1, x2, y2);
        free(crop_buf);
    }

out:
    /* ===== 6. 清理资源 ===== */
    if (src_image.virt_addr) free(src_image.virt_addr);
    release_plate_detector(&det_ctx);
    release_lprnet(&lpr_ctx);
    return 0;
}

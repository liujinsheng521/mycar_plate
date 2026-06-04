#ifndef _PLATE_DETECTOR_H_
#define _PLATE_DETECTOR_H_

#include "rknn_api.h"
#include "common.h"
#include <vector>
#include <string>

/* 单张图片最多检测的车牌数量 */
#define PLATE_OBJ_NUMB_MAX_SIZE 64

/**
 * 车牌检测结果 - 单个边界框
 * 坐标均为原图像素坐标，由模型输出经 letterbox 反算和裁剪得到
 */
typedef struct {
    int left;
    int top;
    int right;
    int bottom;
    float confidence;  /* 检测置信度 (0~1) */
} plate_det_result_t;

/** 车牌检测结果集合 */
typedef struct {
    int count;
    plate_det_result_t results[PLATE_OBJ_NUMB_MAX_SIZE];
} plate_det_results_t;

/**
 * 检测器运行时上下文
 * 保存 RKNN 句柄、输入输出张量属性及模型尺寸，
 * 由 init_plate_detector 填充，后续推理和释放时使用
 */
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} plate_detector_context_t;

/**
 * 初始化车牌检测器
 * @param model_path RKNN 模型文件路径
 * @param ctx        [out] 初始化后的上下文
 * @return 0 成功，-1 失败
 *
 * 步骤：读取模型 → rknn_init 加载到 NPU → 查询输入输出张量属性 → 保存到 ctx
 */
int init_plate_detector(const char *model_path, plate_detector_context_t *ctx);

/**
 * 执行车牌检测推理
 * @param ctx     已初始化的检测器上下文
 * @param src_img 输入图像（RGB888）
 * @param results [out] 检测结果列表
 * @return 0 成功，-1 失败
 *
 * 内部流程：
 *   1. 图像预处理：Letterbox resize + padding 到模型输入尺寸
 *   2. rknn_inputs_set / rknn_run / rknn_outputs_get 调用 NPU
 *   3. 后处理：解析 [N,6] 输出 → 置信度过滤 → 反算原图坐标 → NMS
 */
int inference_plate_detector(plate_detector_context_t *ctx, image_buffer_t *src_img,
                             plate_det_results_t *results);

/**
 * 释放车牌检测器
 * 释放 input_attrs / output_attrs 内存，销毁 RKNN 上下文
 */
int release_plate_detector(plate_detector_context_t *ctx);

#endif

#ifndef _RKNN_DEMO_LPRNET_H_
#define _RKNN_DEMO_LPRNET_H_

#include "rknn_api.h"
#include "common.h"
#include <string>
#include <vector>

/**
 * LPRNet 车牌识别器上下文
 * 保存 RKNN 句柄及模型输入输出属性
 * LPRNet 固定输入尺寸 94x24，BGR 格式
 */
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_width;
    int model_height;
    int model_channel;
} lprnet_context_t;

/** LPRNet 识别结果：车牌字符串，如 "京A12345" */
typedef struct {
    std::string plate_name;
} lprnet_result_t;

/**
 * 初始化 LPRNet
 * @param model_path RKNN 模型路径
 * @param ctx        [out] 初始化后的上下文
 * @return 0 成功，-1 失败
 */
int init_lprnet(const char *model_path, lprnet_context_t *ctx);

/**
 * 执行 LPRNet 车牌字符识别
 * @param ctx    已初始化的 LPRNet 上下文
 * @param img    车牌区域图像（RGB888）
 * @param result [out] 识别结果
 * @return 0 成功，-1 失败
 *
 * 流程：resize 到 94x24 → RGB 转 BGR → NPU 推理 → CTC 贪心解码 → 查字符集拼接车牌
 */
int inference_lprnet(lprnet_context_t *ctx, image_buffer_t *img, lprnet_result_t *result);

/** 释放 LPRNet：销毁 RKNN 上下文，释放内存 */
int release_lprnet(lprnet_context_t *ctx);

#endif

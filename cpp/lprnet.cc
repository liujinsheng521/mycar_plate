#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "lprnet.h"
#include "common.h"
#include "file_utils.h"
#include "opencv2/opencv.hpp"

/**
 * 中国车牌字符集（按 LPRNet 训练索引顺序）
 *   0-30: 省份简称（31 个）
 *  31-40: 数字 0-9
 *  41-65: 字母 A-Z（去掉 I、O）
 *    66:  字母 I
 *    67:  字母 O / CTC blank 符
 */
static const char *CHARS[] = {
    "京","沪","津","渝","冀","晋","蒙","辽","吉","黑",
    "苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤",
    "桂","琼","川","贵","云","藏","陕","甘","青","宁","新",
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","J","K",
    "L","M","N","P","Q","R","S","T","U","V",
    "W","X","Y","Z","I","O","-"};
#define CHARS_NUM 68
#define PLATE_MAX_LEN 18 /* LPRNet 输出序列长度 */

/**
 * 初始化 LPRNet
 *
 * 与 init_plate_detector 流程一致：
 * 读取模型 → rknn_init → 查询输入输出属性 → 解析模型尺寸 → 保存到 ctx
 */
int init_lprnet(const char *model_path, lprnet_context_t *ctx)
{
    int ret;
    char *model;
    rknn_context rknn_ctx = 0;

    int model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load model %s fail!\n", model_path);
        return -1;
    }
    ret = rknn_init(&rknn_ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return -1;

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }

    ctx->rknn_ctx = rknn_ctx;
    ctx->io_num = io_num;
    ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        ctx->model_channel = input_attrs[0].dims[1];
        ctx->model_height  = input_attrs[0].dims[2];
        ctx->model_width   = input_attrs[0].dims[3];
    } else {
        ctx->model_height = input_attrs[0].dims[1];
        ctx->model_width  = input_attrs[0].dims[2];
        ctx->model_channel = input_attrs[0].dims[3];
    }
    return 0;
}

/**
 * 执行 LPRNet 车牌字符识别
 *
 * 预处理：
 *   - resize 到 LPRNet 固定输入 94x24
 *   - RGB → BGR（LPRNet 训练时使用 BGR 格式）
 *
 * NPU 推理：输入 uint8 NHWC，输出 float
 *
 * CTC 贪心解码：
 *   LPRNet 输出形状 [1, 68, 18]，表示 18 个时间步 × 68 个字符类别
 *   1. 每个时间步取 argmax，得到最佳字符索引序列
 *   2. 移除连续重复字符（CTC collapse）
 *   3. 移除 blank 字符（索引 67）
 *   4. 查 CHARS 表拼接为最终车牌字符串
 */
int inference_lprnet(lprnet_context_t *ctx, image_buffer_t *src_img, lprnet_result_t *result)
{
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    /* 预处理：resize 到 94x24，RGB → BGR */
    cv::Mat img_ori(src_img->height, src_img->width, CV_8UC3, src_img->virt_addr);
    cv::Mat img_resized;
    cv::resize(img_ori, img_resized, cv::Size(94, 24));
    cv::cvtColor(img_resized, img_resized, cv::COLOR_RGB2BGR);

    /* NPU 推理 */
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = 94 * 24 * 3;
    inputs[0].buf = img_resized.data;

    ret = rknn_inputs_set(ctx->rknn_ctx, 1, inputs);
    if (ret < 0) return -1;

    ret = rknn_run(ctx->rknn_ctx, nullptr);
    if (ret < 0) return -1;

    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) return -1;

    /* CTC 解码：argmax + collapse + remove blank */
    float *out_data = (float *)outputs[0].buf;
    std::vector<int> no_repeat_blank;

    int prebs[PLATE_MAX_LEN];
    for (int x = 0; x < 18; x++) {
        int max_idx = 0;
        float max_val = -1e10;
        for (int y = 0; y < 68; y++) {
            float val = out_data[y * 18 + x];
            if (val > max_val) { max_val = val; max_idx = y; }
        }
        prebs[x] = max_idx;
    }

    int prev = prebs[0];
    if (prev != 67) no_repeat_blank.push_back(prev);
    for (int i = 0; i < 18; i++) {
        if (prebs[i] == 67 || prebs[i] == prev) {
            if (prebs[i] == 67) prev = prebs[i];
            continue;
        }
        no_repeat_blank.push_back(prebs[i]);
        prev = prebs[i];
    }

    result->plate_name.clear();
    for (int idx : no_repeat_blank) {
        result->plate_name += CHARS[idx];
    }

    rknn_outputs_release(ctx->rknn_ctx, 1, outputs);
    return 0;
}

/** 释放 LPRNet */
int release_lprnet(lprnet_context_t *ctx)
{
    if (ctx->input_attrs)  { free(ctx->input_attrs);  ctx->input_attrs = NULL; }
    if (ctx->output_attrs) { free(ctx->output_attrs); ctx->output_attrs = NULL; }
    if (ctx->rknn_ctx)     { rknn_destroy(ctx->rknn_ctx); ctx->rknn_ctx = 0; }
    return 0;
}

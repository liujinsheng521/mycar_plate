#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "plate_detector.h"
#include "common.h"
#include "file_utils.h"
#include "opencv2/opencv.hpp"

#define OBJ_THRESH  0.5f
#define NMS_THRESH  0.45f

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, "
           "size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims,
           attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size,
           get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static float clamp(float val, float min, float max)
{
    return fmax(fmin(val, max), min);
}

static int nms_boxes(std::vector<plate_det_result_t> &dets, std::vector<int> &keep)
{
    std::vector<int> order(dets.size());
    for (size_t i = 0; i < dets.size(); i++)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return dets[a].confidence > dets[b].confidence;
    });

    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < order.size(); i++)
    {
        int idx = order[i];
        if (suppressed[idx])
            continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < order.size(); j++)
        {
            int idx2 = order[j];
            if (suppressed[idx2])
                continue;

            int x1 = std::max(dets[idx].left, dets[idx2].left);
            int y1 = std::max(dets[idx].top, dets[idx2].top);
            int x2 = std::min(dets[idx].right, dets[idx2].right);
            int y2 = std::min(dets[idx].bottom, dets[idx2].bottom);
            int w = std::max(0, x2 - x1);
            int h = std::max(0, y2 - y1);
            int inter = w * h;
            int area1 = (dets[idx].right - dets[idx].left) *
                        (dets[idx].bottom - dets[idx].top);
            int area2 = (dets[idx2].right - dets[idx2].left) *
                        (dets[idx2].bottom - dets[idx2].top);
            float iou = (float)inter / (area1 + area2 - inter);
            if (iou > NMS_THRESH)
                suppressed[idx2] = true;
        }
    }
    return 0;
}

int init_plate_detector(const char *model_path, plate_detector_context_t *ctx)
{
    int ret;
    char *model;
    rknn_context rknn_ctx = 0;

    int model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load model %s fail!\n", model_path);
        return -1;
    }

    ret = rknn_init(&rknn_ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query IN_OUT_NUM fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query INPUT_ATTR fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query OUTPUT_ATTR fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    ctx->rknn_ctx = rknn_ctx;
    ctx->io_num = io_num;
    ctx->input_attrs = (rknn_tensor_attr *)malloc(
        io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(ctx->input_attrs, input_attrs,
           io_num.n_input * sizeof(rknn_tensor_attr));
    ctx->output_attrs = (rknn_tensor_attr *)malloc(
        io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(ctx->output_attrs, output_attrs,
           io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        ctx->model_channel = input_attrs[0].dims[1];
        ctx->model_height  = input_attrs[0].dims[2];
        ctx->model_width   = input_attrs[0].dims[3];
    }
    else
    {
        ctx->model_height = input_attrs[0].dims[1];
        ctx->model_width  = input_attrs[0].dims[2];
        ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           ctx->model_height, ctx->model_width, ctx->model_channel);

    return 0;
}

int inference_plate_detector(plate_detector_context_t *ctx, image_buffer_t *src_img,
                             plate_det_results_t *results)
{
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];
    int model_w = ctx->model_width;
    int model_h = ctx->model_height;

    memset(results, 0, sizeof(*results));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Convert image_buffer_t to OpenCV Mat (data is RGB888)
    cv::Mat src_mat(src_img->height, src_img->width, CV_8UC3, src_img->virt_addr);

    // Letterbox resize and keep as RGB
    float scale = std::min((float)model_w / src_mat.cols, (float)model_h / src_mat.rows);
    int new_w = (int)(src_mat.cols * scale);
    int new_h = (int)(src_mat.rows * scale);
    int pad_w = (model_w - new_w) / 2;
    int pad_h = (model_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src_mat, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(model_h, model_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));

    // Set input (raw [0,255] values, NHWC)
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = model_w * model_h * ctx->model_channel;
    inputs[0].buf = padded.data;

    ret = rknn_inputs_set(ctx->rknn_ctx, 1, inputs);
    if (ret < 0)
    {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    // Post-process: decode [1, N, 6] format: [cx, cy, w, h, conf, cls]
    float *data = (float *)outputs[0].buf;
    int num_dets = ctx->output_attrs[0].dims[1];

    std::vector<plate_det_result_t> candidates;

    for (int i = 0; i < num_dets; i++)
    {
        float conf = data[i * 6 + 4];
        if (conf < OBJ_THRESH)
            continue;

        float cx = data[i * 6 + 0];
        float cy = data[i * 6 + 1];
        float w  = data[i * 6 + 2];
        float h  = data[i * 6 + 3];

        // Center to corner, remove padding, scale, clip
        float x1 = ((cx - w / 2.0f) - pad_w) / scale;
        float y1 = ((cy - h / 2.0f) - pad_h) / scale;
        float x2 = ((cx + w / 2.0f) - pad_w) / scale;
        float y2 = ((cy + h / 2.0f) - pad_h) / scale;

        x1 = clamp(x1, 0, (float)src_img->width);
        y1 = clamp(y1, 0, (float)src_img->height);
        x2 = clamp(x2, 0, (float)src_img->width);
        y2 = clamp(y2, 0, (float)src_img->height);

        plate_det_result_t det;
        det.left   = (int)x1;
        det.top    = (int)y1;
        det.right  = (int)x2;
        det.bottom = (int)y2;
        det.confidence = conf;
        candidates.push_back(det);
    }

    // NMS
    std::vector<int> keep;
    nms_boxes(candidates, keep);

    results->count = std::min((int)keep.size(), (int)PLATE_OBJ_NUMB_MAX_SIZE);
    for (int i = 0; i < results->count; i++)
        results->results[i] = candidates[keep[i]];

    printf("plate_detector: %zu candidates, %d kept after NMS\n",
           candidates.size(), results->count);

    rknn_outputs_release(ctx->rknn_ctx, 1, outputs);
    return 0;
}

int release_plate_detector(plate_detector_context_t *ctx)
{
    if (ctx->input_attrs != NULL)
    {
        free(ctx->input_attrs);
        ctx->input_attrs = NULL;
    }
    if (ctx->output_attrs != NULL)
    {
        free(ctx->output_attrs);
        ctx->output_attrs = NULL;
    }
    if (ctx->rknn_ctx != 0)
    {
        rknn_destroy(ctx->rknn_ctx);
        ctx->rknn_ctx = 0;
    }
    return 0;
}

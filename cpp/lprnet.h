#ifndef _RKNN_DEMO_LPRNET_H_
#define _RKNN_DEMO_LPRNET_H_

#include "rknn_api.h"
#include "common.h"
#include <string>
#include <vector>

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_width;
    int model_height;
    int model_channel;
} lprnet_context_t;

typedef struct {
    std::string plate_name;
} lprnet_result_t;

int init_lprnet(const char *model_path, lprnet_context_t *ctx);
int inference_lprnet(lprnet_context_t *ctx, image_buffer_t *img, lprnet_result_t *result);
int release_lprnet(lprnet_context_t *ctx);

#endif

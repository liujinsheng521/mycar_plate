#ifndef _PLATE_DETECTOR_H_
#define _PLATE_DETECTOR_H_

#include "rknn_api.h"
#include "common.h"
#include <vector>
#include <string>

#define PLATE_OBJ_NUMB_MAX_SIZE 64

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
    float confidence;
} plate_det_result_t;

typedef struct {
    int count;
    plate_det_result_t results[PLATE_OBJ_NUMB_MAX_SIZE];
} plate_det_results_t;

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} plate_detector_context_t;

int init_plate_detector(const char *model_path, plate_detector_context_t *ctx);
int inference_plate_detector(plate_detector_context_t *ctx, image_buffer_t *src_img,
                             plate_det_results_t *results);
int release_plate_detector(plate_detector_context_t *ctx);

#endif // _PLATE_DETECTOR_H_

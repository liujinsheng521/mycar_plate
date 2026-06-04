/*
 * License Plate Detection + Recognition Demo
 * Usage: ./rknn_plate_detection_demo <detect_model.rknn> <lprnet_model.rknn> <image.jpg>
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

    // Init detector
    printf(">>> Init plate detector: %s\n", det_model_path);
    plate_detector_context_t det_ctx;
    memset(&det_ctx, 0, sizeof(det_ctx));
    int ret = init_plate_detector(det_model_path, &det_ctx);
    if (ret != 0) return -1;

    // Init LPRNet
    printf(">>> Init LPRNet: %s\n", lpr_model_path);
    lprnet_context_t lpr_ctx;
    memset(&lpr_ctx, 0, sizeof(lpr_ctx));
    ret = init_lprnet(lpr_model_path, &lpr_ctx);
    if (ret != 0) { release_plate_detector(&det_ctx); return -1; }

    // Read image
    printf(">>> Read image: %s\n", image_path);
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0) { printf("read_image fail!\n"); goto out; }
    printf("  image size: %dx%d\n", src_image.width, src_image.height);

    // Detect
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
    for (int i = 0; i < det_results.count; i++)
    {
        plate_det_result_t &det = det_results.results[i];

        int x1 = det.left < 0 ? 0 : det.left;
        int y1 = det.top < 0 ? 0 : det.top;
        int x2 = det.right > src_image.width ? src_image.width : det.right;
        int y2 = det.bottom > src_image.height ? src_image.height : det.bottom;
        if (x2 <= x1 || y2 <= y1) continue;

        int crop_w = x2 - x1;
        int crop_h = y2 - y1;
        if (crop_w * crop_h < src_image.width * src_image.height * 0.0005) continue;

        // Copy crop region to contiguous buffer (original rows may be wider)
        unsigned char *crop_buf = (unsigned char *)malloc(crop_w * crop_h * 3);
        if (!crop_buf) continue;
        unsigned char *src_row = src_image.virt_addr + (y1 * src_image.width + x1) * 3;
        for (int r = 0; r < crop_h; r++) {
            memcpy(crop_buf + r * crop_w * 3, src_row + r * src_image.width * 3, crop_w * 3);
        }

        image_buffer_t plate_img;
        memset(&plate_img, 0, sizeof(plate_img));
        plate_img.width = crop_w;
        plate_img.height = crop_h;
        plate_img.format = IMAGE_FORMAT_RGB888;
        plate_img.size = crop_w * crop_h * 3;
        plate_img.virt_addr = crop_buf;

        lprnet_result_t lpr_result;
        ret = inference_lprnet(&lpr_ctx, &plate_img, &lpr_result);
        if (ret != 0) { free(crop_buf); continue; }

        printf("  [%d] Plate: %s (conf: %.3f)  box: (%d,%d,%d,%d)\n",
               i, lpr_result.plate_name.c_str(), det.confidence,
               x1, y1, x2, y2);
        free(crop_buf);
    }

out:
    if (src_image.virt_addr) free(src_image.virt_addr);
    release_plate_detector(&det_ctx);
    release_lprnet(&lpr_ctx);
    return 0;
}

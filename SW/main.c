#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "layer.h"

#define RGB 3
#define INPUT_H 227
#define INPUT_W 227
#define P(x) x*x

#define LIN
#define fwriteIWB
#define LOUT

int rand_int4() {
    return (rand() % 16) - 8;
}

void rand_fill(int8_t* buf, int size) {
    for (int i = 0; i < size; ++i)
        buf[i] = rand_int4();
}

int main() {
    srand(42);

    // Input
    int8_t* input = malloc(RGB * INPUT_H * INPUT_W);

    // Conv1: 3x11x11 → 96x55x55
    int8_t* conv1_w = malloc(96 * 3 * 11 * 11);
    int8_t* conv1_b = malloc(96);
    int8_t* conv1_out = malloc(96 * P(55));

    // Conv2: 96x5x5 → 256x27x27
    int8_t* conv2_w = malloc(256 * 96 * 5 * 5);
    int8_t* conv2_b = malloc(256);
    int8_t* conv2_out = malloc(256 * P(27));

    // Conv3: 256x3x3 → 384x13x13
    int8_t* conv3_w = malloc(384 * 256 * 3 * 3);
    int8_t* conv3_b = malloc(384);
    int8_t* conv3_out = malloc(384 * P(13));

    // Conv4: 384x3x3 → 384x13x13
    int8_t* conv4_w = malloc(384 * 384 * 3 * 3);
    int8_t* conv4_b = malloc(384);
    int8_t* conv4_out = malloc(384 * P(13));

    // Conv5: 384x3x3 → 256x13x13
    int8_t* conv5_w = malloc(256 * 384 * 3 * 3);
    int8_t* conv5_b = malloc(256);
    int8_t* conv5_out = malloc(256 * P(13));

    // Pool5 → 256 x 6 x 6 = 9216
    int8_t* pool5_out = malloc(256 * P(6));

    // FC1: 9216 → 4096
    int8_t* fc1_w = malloc(9216 * 4096);
    int8_t* fc1_b = malloc(4096);
    int8_t* fc1_out = malloc(4096);

    // FC2: 4096 → 4096
    int8_t* fc2_w = malloc(4096 * 4096);
    int8_t* fc2_b = malloc(4096);
    int8_t* fc2_out = malloc(4096);

    // FC3: 4096 → 1000
    int8_t* fc3_w = malloc(4096 * 1000);
    int8_t* fc3_b = malloc(1000);
    int8_t* fc3_out = malloc(1000); // softmax 결과는 float

    // ---------------------------
    // 값 초기화
    rand_fill(input, RGB * INPUT_H * INPUT_W);
    rand_fill(conv1_w, 96 * 3 * 11 * 11); rand_fill(conv1_b, 96);
    rand_fill(conv2_w, 256 * 96 * 5 * 5); rand_fill(conv2_b, 256);
    rand_fill(conv3_w, 384 * 256 * 3 * 3); rand_fill(conv3_b, 384);
    rand_fill(conv4_w, 384 * 384 * 3 * 3); rand_fill(conv4_b, 384);
    rand_fill(conv5_w, 256 * 384 * 3 * 3); rand_fill(conv5_b, 256);
    rand_fill(fc1_w, 9216 * 4096); rand_fill(fc1_b, 4096);
    rand_fill(fc2_w, 4096 * 4096); rand_fill(fc2_b, 4096);
    rand_fill(fc3_w, 4096 * 1000); rand_fill(fc3_b, 1000);

#ifdef LIN
    write_output_data("input", input, RGB * INPUT_H * INPUT_W);
#endif

#ifdef fwriteIWB
    write_layer_data("conv1", conv1_w, 96 * 3 * 11 * 11, conv1_b, 96);
    write_layer_data("conv2", conv2_w, 256 * 96 * 5 * 5, conv2_b, 256);
    write_layer_data("conv3", conv3_w, 384 * 256 * 3 * 3, conv3_b, 384);
    write_layer_data("conv4", conv4_w, 384 * 384 * 3 * 3, conv4_b, 384);
    write_layer_data("conv5", conv5_w, 256 * 384 * 3 * 3, conv5_b, 256);

    write_layer_data("fc1", fc1_w, 9216 * 4096, fc1_b, 4096);
    write_layer_data("fc2", fc2_w, 4096 * 4096, fc2_b, 4096);
    write_layer_data("fc3", fc3_w, 4096 * 1000, fc3_b, 1000);
#endif

    // ---------------------------
    // Conv1 → ReLU → Pool
    conv2d_multi(input, conv1_out, conv1_w, conv1_b, 3, 96, 227, 227, 11, 4, 0);
    relu(conv1_out, 96 * 55 * 55);
    maxpool2d(conv1_out, conv2_out, 96, 55, 55, 3, 2);  // output: 96x27x27
    #ifdef LOUT
        write_output_data("conv1_out", conv1_out, 96*27*27);
    #endif

    // Conv2 → ReLU → Pool
    conv2d_multi(conv2_out, conv2_out, conv2_w, conv2_b, 96, 256, 27, 27, 5, 1, 2);
    relu(conv2_out, 256 * 27 * 27);
    maxpool2d(conv2_out, conv3_out, 256, 27, 27, 3, 2); // output: 256x13x13
    #ifdef LOUT
        write_output_data("conv2_out", conv2_out, 256*13*13);
    #endif

    // Conv3 → ReLU
    conv2d_multi(conv3_out, conv3_out, conv3_w, conv3_b, 256, 384, 13, 13, 3, 1, 1);
    relu(conv3_out, 384 * 13 * 13);
    #ifdef LOUT
        write_output_data("conv3_out", conv3_out, 96*27*27);
    #endif

    // Conv4 → ReLU
    conv2d_multi(conv3_out, conv4_out, conv4_w, conv4_b, 384, 384, 13, 13, 3, 1, 1);
    relu(conv4_out, 384 * 13 * 13);
    #ifdef LOUT
        write_output_data("conv4_out", conv4_out, 96*27*27);
    #endif


    // Conv5 → ReLU → Pool
    conv2d_multi(conv4_out, conv5_out, conv5_w, conv5_b, 384, 256, 13, 13, 3, 1, 1);
    relu(conv5_out, 256 * 13 * 13);
    // void maxpool2d(int8_t* input, int8_t* output, int in_w, int in_h, int ksize, int stride)
    maxpool2d(conv5_out, pool5_out, 256, 13, 13, 3, 2); // output: 256x6x6 → flat: 9216
    #ifdef LOUT
        write_output_data("conv5_out", conv5_out, 256*6*6);
    #endif

    // FC1 → ReLU
    fc_layer(pool5_out, fc1_out, fc1_w, fc1_b, 9216, 4096);
    relu(fc1_out, 4096);
    #ifdef LOUT
        write_output_data("fc1_out", fc1_out, 4096);
    #endif

    // FC2 → ReLU
    fc_layer(fc1_out, fc2_out, fc2_w, fc2_b, 4096, 4096);
    relu(fc2_out, 4096);
    #ifdef LOUT
        write_output_data("fc2_out", fc2_out, 4096);
    #endif

    // FC3 → Softmax
    fc_layer(fc2_out, fc3_out, fc3_w, fc3_b, 4096, 1000);  // casting to reuse fc3_out buffer
    #ifdef LOUT
        write_output_data("fc3_out", fc3_out, 1000);
    #endif
    softmax_int8(fc3_out, (float *)fc3_out, 1000);  // 최종 소프트맥스 출력
    // Softmax (출력은 float 배열에)
    // ---------------------------

    // printf("Softmax Output Sample: %f\n", fc3_out[0]);
    #ifdef LOUT
        write_output("model_out", (float *)fc3_out, 1000);
    #endif

    // ---------------------------
    // 해제
    free(input);
    free(conv1_w); free(conv1_b); free(conv1_out);
    free(conv2_w); free(conv2_b); free(conv2_out);
    free(conv3_w); free(conv3_b); free(conv3_out);
    free(conv4_w); free(conv4_b); free(conv4_out);
    free(conv5_w); free(conv5_b); free(conv5_out);
    free(pool5_out);
    free(fc1_w); free(fc1_b); free(fc1_out);
    free(fc2_w); free(fc2_b); free(fc2_out);
    free(fc3_w); free(fc3_b); free(fc3_out);

    return 0;
}

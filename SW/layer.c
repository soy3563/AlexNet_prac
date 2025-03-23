#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define STRIDE 1

// 간단한 2D convolution: 단일 채널 기준, 패딩 없음
void conv2d_multi(int8_t* input, int8_t* output, int8_t* kernel, int8_t* bias,
    int in_c, int out_c, int in_w, int in_h, int ksize, int stride, int pad) {

    int out_w = (in_w - ksize + 2 * pad) / stride + 1;
    int out_h = (in_h - ksize + 2 * pad) / stride + 1;

    for (int oc = 0; oc < out_c; ++oc) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                    int sum = 0;
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int ky = 0; ky < ksize; ++ky) {
                            for (int kx = 0; kx < ksize; ++kx) {
                                int ix = ox * stride + kx - pad;
                                int iy = oy * stride + ky - pad;
                                if (ix >= 0 && iy >= 0 && ix < in_w && iy < in_h) {
                                    int in_idx = ic * in_h * in_w + iy * in_w + ix;
                                    int k_idx = oc * in_c * ksize * ksize + ic * ksize * ksize + ky * ksize + kx;
                                    sum += input[in_idx] * kernel[k_idx];
                                }
                            }
                        }
                    }
                    int out_idx = oc * out_h * out_w + oy * out_w + ox;
                    output[out_idx] = sum + bias[oc];
                }
            }
        }
}

// ReLU 함수
void relu(int8_t* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = fmaxf(0, data[i]);
}

// MaxPooling (3x3, stride 2)
void maxpool2d(int8_t* input, int8_t* output, int in_c, int in_w, int in_h,
    int ksize, int stride) {

    int out_w = (in_w - ksize) / stride + 1;
    int out_h = (in_h - ksize) / stride + 1;

    for (int c = 0; c < in_c; ++c) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                int8_t max_val = INT8_MIN;
                for (int ky = 0; ky < ksize; ++ky) {
                    for (int kx = 0; kx < ksize; ++kx) {
                        int ix = x * stride + kx;
                        int iy = y * stride + ky;
                        int in_idx = c * in_h * in_w + iy * in_w + ix;
                        if (input[in_idx] > max_val)
                            max_val = input[in_idx];
                    }
                }
                int out_idx = c * out_h * out_w + y * out_w + x;
                output[out_idx] = max_val;
            }
        }
    }
}

// FC 레이어
void fc_layer(int8_t* input, int8_t* output, int8_t* weight, int8_t* bias, int in_size, int out_size) {
    for (int i = 0; i < out_size; ++i) {
        int8_t sum = bias[i];
        for (int j = 0; j < in_size; ++j) {
            sum += input[j] * weight[i * in_size + j];
        }
        output[i] = sum;
    }
}

void softmax_int8(int8_t* input, float* output, int size) {
    float max = (float)input[0];
    for (int i = 1; i < size; ++i)
        if ((float)input[i] > max) max = (float)input[i];

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = expf((float)input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i)
        output[i] /= sum;
}

void write_layer_data(const char* layer_name, int8_t* weight, int weight_size, int8_t* bias, int bias_size) {
    // char wfile[64], bfile[64], ifile[64];
    char wfile[64], bfile[64];
    snprintf(wfile, sizeof(wfile), "%s_weight.txt", layer_name);
    snprintf(bfile, sizeof(bfile), "%s_bias.txt", layer_name);
    // snprintf(ifile, sizeof(ifile), "%s_input.txt", layer_name);

    FILE* fw = fopen(wfile, "w");
    FILE* fb = fopen(bfile, "w");
    // FILE* fi = fopen(ifile, "w");

    if (fw && fb) {
    // if (fw && fb && fi) {
    for (int i = 0; i < weight_size; ++i)
    fprintf(fw, "%d\n", weight[i]);

    for (int i = 0; i < bias_size; ++i)
    fprintf(fb, "%d\n", bias[i]);

    // for (int i = 0; i < in_size; ++i)
    // fprintf(fi, "%d\n", input[i]);
    }

    if (fw) fclose(fw);
    if (fb) fclose(fb);
    // if (fi) fclose(fi);
}

void write_output_data(const char* layer_name, int8_t* output, int size) {
    char fname[64];
    snprintf(fname, sizeof(fname), "%s_out.txt", layer_name);

    FILE* f = fopen(fname, "w");
    if (!f) return;

    for (int i = 0; i < size; ++i)
        fprintf(f, "%d\n", output[i]);

    fclose(f);
}

void write_output(const char* layer_name, float* output, int size) {
    char fname[64];
    snprintf(fname, sizeof(fname), "%s_out.txt", layer_name);

    FILE* f = fopen(fname, "w");
    if (!f) return;

    for (int i = 0; i < size; ++i)
        fprintf(f, "%f\n", output[i]);

    fclose(f);
}
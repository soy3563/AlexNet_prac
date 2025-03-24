#ifndef LAYER_H
#define LAYER_H

void conv2d_multi(int8_t* input, int8_t* output, int8_t* weight, int8_t* bias,
    int in_c, int out_c, int in_w, int in_h, int ksize, int stride, int pad);

void relu(int8_t* data, int size);

void maxpool2d(int8_t* input, int8_t* output, int in_c, int in_w, int in_h, int ksize, int stride);

void fc_layer(int8_t* input, int8_t* output, int8_t* weight, int8_t* bias, int in_size, int out_size);

void softmax(float* input, float* output, int size);

void softmax_int8(int8_t* input, float* output, int size);

void write_layer_data(const char* layer_name, int8_t* weight, int weight_size, int8_t* bias, int bias_size);

// void write_layer_data(const char* layer_name, int8_t* input, int in_size,
//     int8_t* weight, int weight_size, int8_t* bias, int bias_size);

void write_output_data(const char* layer_name, int8_t* output, int size);

void write_output(const char* layer_name, float* output, int size);

#endif

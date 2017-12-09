#ifndef IM2COL_H
#define IM2COL_H
#include "darknet.h"

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);


void im2col_cpu_h(int16_t* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, int16_t* data_col);

int* im2col_id_h(int channels,
                 int height, int width,
                 int ksize,
                 int stride,
                 int pad);



#endif

#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

int16_t im2col_get_pixel_h(int16_t *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

int* im2col_id_h(int channels,
                 int height, int width,
                 int ksize,
                 int stride,
                 int pad)

{
  int height_col = (height + 2*pad - ksize) / stride + 1;
  int width_col = (width + 2*pad - ksize) / stride + 1;
  int channel_size = height*width;
  int output_shape = height_col * width_col * ksize * ksize * channels;

  int* id = calloc (output_shape, sizeof(int));

  int* col_id_it = id;
  int im_ptr = 0;

  for (int channel = channels; channel--; im_ptr += channel_size) {
    for (int kernel_row = 0; kernel_row < ksize; kernel_row++) {
      for (int kernel_col = 0; kernel_col < ksize; kernel_col++) {
        int input_row = kernel_row - pad;
        for (int output_row = height_col; output_row; output_row--) {
          if (input_row < 0 || input_row >= height) {
            for (int output_col = width_col; output_col; output_col--) {
              *(col_id_it++) = -1;
            }
          } else {
            int input_col = -pad + kernel_col;
            for (int output_col = width_col; output_col; output_col--) {
              if (input_col >= 0 && input_col < width) {
                *(col_id_it++) = (im_ptr + input_row*width + input_col)*sizeof(int16_t);
              } else {
                *(col_id_it++) = -1;
              }
              input_col += stride;
            }
          }
          input_row += stride;
        }
      }
    }
  }
  return id;
}

void im2col_cpu_h(int16_t* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, int16_t* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_h(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

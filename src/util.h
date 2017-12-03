#ifndef HWACHA_UTIL_H
#define HWACHA_UTIL_H

#define VRU_ENABLE

#ifdef VRU_ENABLE
// because gcc complains about shifting without L
#define VRU_SWITCH 0x8000000000000000
#else
#define VRU_SWITCH 0x0
#endif



#define VCFG(nvvd, nvvw, nvvh, nvp) \
  (((nvvd) & 0x1ff) | \
  (((nvp) & 0x1f) << 9) | \
  (((nvvw) & 0x1ff) << 14) | \
  (((nvvh) & 0x1ff) << 23) | \
  (VRU_SWITCH))

int rdcycle();
int rdinstret();
void* safe_malloc(int size);
void printfloatmatrix(int channels, int width, int height, float* M);
void printintmatrix(int channels, int width, int height, int* M);
void fill(float* p, int n, int mode);
void setvcfg(int nd, int nw, int nh, int np);
int setvlen(int vlen);
void vec_gather(const int* id, const float* src, float* dest, int len);
void acc_gather(float* dest, int len, int stride, float* src, int* ids, int n_iter);

#endif

#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <stdio.h>
#include "util.h"



int __attribute__((optimize("O0"))) rdcycle() {
    int out = 0;
    asm("rdcycle %0" : "=r" (out));
    return out;
}

int __attribute__((optimize("O0"))) rdinstret() {
    int out = 0;
    asm("rdinstret %0" : "=r" (out));
    return out;
}

void* __attribute__((optimize("O0"))) safe_malloc(int size) {
    void* ptr = memalign(16, size);
    for (int i = 0; i < size / 4; i += (1 << 10)) {
        ((int*)ptr)[i] = 1;
    }
    return ptr;
}

void printfloatmatrix(int channels, int width, int height, float* M) {
    printf("\n");
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%.3f\t", M[c*height*width+i*width+j]);
            }
            printf("\n");
        }
        printf("-----\n");
    }
}
void printintmatrix(int channels, int width, int height, int* M) {
    printf("\n");
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%d\t", M[c*height*width+i*width+j]);
            }
            printf("\n");
        }
        printf("-----\n");
    }
}

void fill(float* p, int n, int mode) {
    for (int i = 0; i < n; i++) {
        if (mode == 0) {
            p[i] = i;
        } else if (mode == 1) {
            p[i] = (float)rand() / (float)(RAND_MAX);
        } else if (mode == 2) {
            p[i] = 1;
        }
    }
}

void setvcfg(int nd, int nw, int nh, int np) {
    int cfg = VCFG(nd, nw, nh, np);
    asm volatile ("vsetcfg %0"
                  :
                  : "r" (cfg));
}

int setvlen(int vlen) {
    int consumed;
    asm volatile ("vsetvl %0, %1"
                  : "=r" (consumed)
                  : "r" (vlen));
    return consumed;
}

void vec_gather(const int* id, const float* src, float* dest, int len) {
    setvcfg(0, 2, 0, 2);
    for (int i = 0; i < len; ) {
        int consumed = setvlen(len - i);

        asm volatile ("vmcs vs1, %0"
                      :
                      : "r" (&src[0]));
        asm volatile ("vmca va1, %0"
                      :
                      : "r" (&id[i]));
        asm volatile ("vmca va2, %0"
                      :
                      : "r" (&dest[i]));
        asm volatile ("la t0, vgather"
                      :
                      :
                      : "t0");
        asm volatile ("vf 0(t0)");
        
        i += consumed;
    }
    asm volatile ("fence");
}

void acc_gather(float* dest, int len, int stride, float* src,
                int* ids, int n_iter) {
    setvcfg(0, 3, 0, 1);
    int cycles = rdcycle();
    int z = 0;
    for (int j = 0; j < len; ) {
        int consumed = setvlen(len - j);

        asm volatile ("vmca va0, %0"
                      :
                      : "r" (&ids[j]));
        asm volatile ("la t0, vacc_gather_pre"
                      :
                      :
                      : "t0");
        asm volatile ("vf 0(t0)");
        for (int i = 0; i < n_iter; i++) {
            asm volatile ("vmcs vs1, %0"
                          :
                          : "r" (&src[i*stride]));
            asm volatile ("la t0, vacc_gather"
                          :
                          :
                          : "t0");
            asm volatile ("vf 0(t0)");
        }
        asm volatile ("vmca va0, %0"
                      :
                      : "r" (&dest[j]));
        asm volatile ("la t0, vacc_gather_post"
                      :
                      :
                      : "t0");
        asm volatile ("vf 0(t0)");
        j += consumed;
        z++;
    }
    asm volatile ("fence");
    printf("%d %d\n", rdcycle() - cycles, z);

}

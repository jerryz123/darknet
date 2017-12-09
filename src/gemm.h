#ifndef GEMM_H
#define GEMM_H
#include <stdint.h>
void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_h(int TA, int TB, int M, int N, int K, float ALPHA, 
                    int16_t *A, int lda, 
                    int16_t *B, int ldb,
                    float BETA,
                    int16_t *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

void gemm_cpu_h(int TA, int TB, int M, int N, int K, float ALPHA, 
        int16_t *A, int lda, 
        int16_t *B, int ldb,
        float BETA,
        int16_t *C, int ldc);


#endif

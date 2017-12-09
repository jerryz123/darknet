#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "util.h"

void gemm_bin(int M, int N, int K, float ALPHA,
        char  *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_h(int TA, int TB, int M, int N, int K, float ALPHA,
        int16_t *A, int lda,
        int16_t *B, int ldb,
        float BETA,
        int16_t *C, int ldc)
{
    gemm_cpu_h( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}


void gemm_nn_hwacha(int M, int N, int K,
                      float ALPHA,
                      float* A, int lda,
                      float* B, int ldb,
                      float* C, int ldc)
{
   asm volatile ("vsetcfg %0" : : "r" (VCFG(0, 8, 0, 1)));

   int vlen_result;
   asm volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (N));

   void * vpset_vfblockaddr;
   asm volatile ("la %0, sgemm_opt_v_4_4_vpset" : "=r" (vpset_vfblockaddr));

   asm volatile ("vf 0(%0)" : : "r" (vpset_vfblockaddr));

   void * pre_vfblockaddr;
   void * pre_edge_vfblockaddr;
   void * main_vfblockaddr;
   void * main_edge0_vfblockaddr;
   void * main_edge1_vfblockaddr;
   void * post_vfblockaddr;
   void * post_edge_vfblockaddr;
   asm volatile ("la %0, sgemm_opt_v_4_4_pre" : "=r" (pre_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (pre_vfblockaddr) : "t0");
   asm volatile ("la %0, sgemm_opt_v_4_4_pre_edge" : "=r" (pre_edge_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (pre_edge_vfblockaddr) : "t0");
   asm volatile ("la %0, sgemm_opt_v_4_4" : "=r" (main_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (main_vfblockaddr) : "t0");
   asm volatile ("la %0, sgemm_opt_v_4_4_edge0" : "=r" (main_edge0_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (main_edge0_vfblockaddr) : "t0");
   asm volatile ("la %0, sgemm_opt_v_4_4_edge1" : "=r" (main_edge1_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (main_edge1_vfblockaddr) : "t0");
   asm volatile ("la %0, sgemm_opt_v_4_4_post" : "=r" (post_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (post_vfblockaddr) : "t0");
   asm volatile ("la %0, sgemm_opt_v_4_4_post_edge" : "=r" (post_edge_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (post_edge_vfblockaddr) : "t0");
   int i;
   for (i = 0; i + 4 <= M; i+=4) {
     for (int k = 0; k < N; ) {
       int consumed;
       int artificial = N - k;

       asm volatile ("vsetvl %0, %1" : "=r" (consumed) : "r" (artificial));

       // C rows 1, 2, 3, 4
       asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));
       asm volatile ("vmca va1, %0" : : "r" (&C[(i+1)*ldc+k]));
       asm volatile ("vmca va2, %0" : : "r" (&C[(i+2)*ldc+k]));
       asm volatile ("vmca va3, %0" : : "r" (&C[(i+3)*ldc+k]));

       asm volatile ("vf 0(%0)" : : "r" (pre_vfblockaddr));
       int j;
       for (j = 0; j + 4 <= K; j+=4) {

         // B row 1, 2, 3, 4
         asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
         asm volatile ("vmca va5, %0" : : "r" (&B[(j+1)*ldb+k]));
         asm volatile ("vmca va6, %0" : : "r" (&B[(j+2)*ldb+k]));
         asm volatile ("vmca va7, %0" : : "r" (&B[(j+3)*ldb+k]));

         // A row 1, 2, 3, 4
         asm volatile ("vmcs vs1, %0\n"
                       "vmcs vs2, %1\n"
                       "vmcs vs3, %2\n"
                       "vmcs vs4, %3\n"

                       "vmcs vs5, %4\n"
                       "vmcs vs6, %5\n"
                       "vmcs vs7, %6\n"
                       "vmcs vs8, %7\n"

                       "vmcs vs9, %8\n"
                       "vmcs vs10, %9\n"
                       "vmcs vs11, %10\n"
                       "vmcs vs12, %11\n"

                       "vmcs vs13, %12\n"
                       "vmcs vs14, %13\n"
                       "vmcs vs15, %14\n"
                       "vmcs vs16, %15"
                       :
                       : "r" (A[j+(i+0)*lda+0]), "r" (A[j+(i+0)*lda+1]), "r" (A[j+(i+0)*lda+2]), "r" (A[j+(i+0)*lda+3]),
                         "r" (A[j+(i+1)*lda+0]), "r" (A[j+(i+1)*lda+1]), "r" (A[j+(i+1)*lda+2]), "r" (A[j+(i+1)*lda+3]),
                         "r" (A[j+(i+2)*lda+0]), "r" (A[j+(i+2)*lda+1]), "r" (A[j+(i+2)*lda+2]), "r" (A[j+(i+2)*lda+3]),
                         "r" (A[j+(i+3)*lda+0]), "r" (A[j+(i+3)*lda+1]), "r" (A[j+(i+3)*lda+2]), "r" (A[j+(i+3)*lda+3])
                       );

         asm volatile ("vf 0(%0)" : : "r" (main_vfblockaddr));
       }

       for ( ; j < K; j++) {
         asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));

         asm volatile ("vmcs vs1, %0\n"

                       "vmcs vs5, %1\n"

                       "vmcs vs9, %2\n"

                       "vmcs vs13, %3\n"
                       :
                       : "r" (A[j+(i+0)*lda+0]),
                         "r" (A[j+(i+1)*lda+0]),
                         "r" (A[j+(i+2)*lda+0]),
                         "r" (A[j+(i+3)*lda+0])
                       );
         asm volatile ("vf 0(%0)" : : "r" (main_edge0_vfblockaddr));

       }
       asm volatile ("vf 0(%0)" : : "r" (post_vfblockaddr));
       k += consumed;
     }
   }

   for ( ; i < M; i++) {
     for (int k = 0; k < N; ) {
       int consumed;
       int artificial = N - k;

       asm volatile ("vsetvl %0, %1" : "=r" (consumed) : "r" (artificial));
       asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));

       asm volatile ("vf 0(%0)" : : "r" (pre_edge_vfblockaddr));

       for (int j = 0; j < K; j++) {
         asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
         asm volatile ("vmcs vs1, %0" : : "r" (A[j+(i+0)*lda+0]));
         asm volatile ("vf 0(%0)" : : "r" (main_edge1_vfblockaddr));
       }
       asm volatile ("vf 0(%0)" : : "r" (post_edge_vfblockaddr));
       k += consumed;
     }
   }
   if (ALPHA > 1.01 || ALPHA < 0.99)
     scalex(C, ALPHA, M*N);
   asm volatile ("fence");
}

void gemm_nn_hwacha_h(int M, int N, int K,
                      float ALPHA,
                      int16_t* A, int lda,
                      int16_t* B, int ldb,
                      int16_t* C, int ldc)
{
   asm volatile ("vsetcfg %0" : : "r" (VCFG(0, 0, 8, 1)));

   int vlen_result;
   asm volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (N));

   void * vpset_vfblockaddr;
   asm volatile ("la %0, hgemm_opt_v_4_4_vpset" : "=r" (vpset_vfblockaddr));
   asm volatile ("vf 0(%0)" : : "r" (vpset_vfblockaddr));

   void * pre_vfblockaddr;
   void * pre_edge_vfblockaddr;
   void * main_vfblockaddr;
   void * main_edge0_vfblockaddr;
   void * main_edge1_vfblockaddr;
   void * post_vfblockaddr;
   void * post_edge_vfblockaddr;
   asm volatile ("la %0, hgemm_opt_v_4_4_pre" : "=r" (pre_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (pre_vfblockaddr) : "t0");
   asm volatile ("la %0, hgemm_opt_v_4_4_pre_edge" : "=r" (pre_edge_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (pre_edge_vfblockaddr) : "t0");
   asm volatile ("la %0, hgemm_opt_v_4_4" : "=r" (main_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (main_vfblockaddr) : "t0");
   asm volatile ("la %0, hgemm_opt_v_4_4_edge0" : "=r" (main_edge0_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (main_edge0_vfblockaddr) : "t0");
   asm volatile ("la %0, hgemm_opt_v_4_4_edge1" : "=r" (main_edge1_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (main_edge1_vfblockaddr) : "t0");
   asm volatile ("la %0, hgemm_opt_v_4_4_post" : "=r" (post_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (post_vfblockaddr) : "t0");
   asm volatile ("la %0, hgemm_opt_v_4_4_post_edge" : "=r" (post_edge_vfblockaddr));
   asm volatile ("lw t0, 0(%0)" : : "r" (post_edge_vfblockaddr) : "t0");
   int i;
   for (i = 0; i + 4 <= M; i+=4) {
     for (int k = 0; k < N; ) {
       int consumed;
       int artificial = N - k;

       asm volatile ("vsetvl %0, %1" : "=r" (consumed) : "r" (artificial));
       // C rows 1, 2, 3, 4
       asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));
       asm volatile ("vmca va1, %0" : : "r" (&C[(i+1)*ldc+k]));
       asm volatile ("vmca va2, %0" : : "r" (&C[(i+2)*ldc+k]));
       asm volatile ("vmca va3, %0" : : "r" (&C[(i+3)*ldc+k]));
       asm volatile ("vf 0(%0)" : : "r" (pre_vfblockaddr));
       int j;
       for (j = 0; j + 4 <= K; j+=4) {

         // B row 1, 2, 3, 4
         asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
         asm volatile ("vmca va5, %0" : : "r" (&B[(j+1)*ldb+k]));
         asm volatile ("vmca va6, %0" : : "r" (&B[(j+2)*ldb+k]));
         asm volatile ("vmca va7, %0" : : "r" (&B[(j+3)*ldb+k]));

         // A row 1, 2, 3, 4
         asm volatile ("vmcs vs1, %0\n"
                       "vmcs vs2, %1\n"
                       "vmcs vs3, %2\n"
                       "vmcs vs4, %3\n"

                       "vmcs vs5, %4\n"
                       "vmcs vs6, %5\n"
                       "vmcs vs7, %6\n"
                       "vmcs vs8, %7\n"

                       "vmcs vs9, %8\n"
                       "vmcs vs10, %9\n"
                       "vmcs vs11, %10\n"
                       "vmcs vs12, %11\n"

                       "vmcs vs13, %12\n"
                       "vmcs vs14, %13\n"
                       "vmcs vs15, %14\n"
                       "vmcs vs16, %15"
                       :
                       : "r" (A[j+(i+0)*lda+0]), "r" (A[j+(i+0)*lda+1]), "r" (A[j+(i+0)*lda+2]), "r" (A[j+(i+0)*lda+3]),
                         "r" (A[j+(i+1)*lda+0]), "r" (A[j+(i+1)*lda+1]), "r" (A[j+(i+1)*lda+2]), "r" (A[j+(i+1)*lda+3]),
                         "r" (A[j+(i+2)*lda+0]), "r" (A[j+(i+2)*lda+1]), "r" (A[j+(i+2)*lda+2]), "r" (A[j+(i+2)*lda+3]),
                         "r" (A[j+(i+3)*lda+0]), "r" (A[j+(i+3)*lda+1]), "r" (A[j+(i+3)*lda+2]), "r" (A[j+(i+3)*lda+3])
                       );

         asm volatile ("vf 0(%0)" : : "r" (main_vfblockaddr));
       }

       for ( ; j < K; j++) {
         asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));

         asm volatile ("vmcs vs1, %0\n"

                       "vmcs vs5, %1\n"

                       "vmcs vs9, %2\n"

                       "vmcs vs13, %3\n"
                       :
                       : "r" (A[j+(i+0)*lda+0]),
                         "r" (A[j+(i+1)*lda+0]),
                         "r" (A[j+(i+2)*lda+0]),
                         "r" (A[j+(i+3)*lda+0])
                       );
         asm volatile ("vf 0(%0)" : : "r" (main_edge0_vfblockaddr));

       }
       asm volatile ("vf 0(%0)" : : "r" (post_vfblockaddr));
       k += consumed;
     }
   }

   for ( ; i < M; i++) {
     for (int k = 0; k < N; ) {
       int consumed;
       int artificial = N - k;

       asm volatile ("vsetvl %0, %1" : "=r" (consumed) : "r" (artificial));
       asm volatile ("vmca va0, %0" : : "r" (&C[i*ldc+k]));

       asm volatile ("vf 0(%0)" : : "r" (pre_edge_vfblockaddr));

       for (int j = 0; j < K; j++) {
         asm volatile ("vmca va4, %0" : : "r" (&B[j*ldb+k]));
         asm volatile ("vmcs vs1, %0" : : "r" (A[j+(i+0)*lda+0]));
         asm volatile ("vf 0(%0)" : : "r" (main_edge1_vfblockaddr));
       }
       asm volatile ("vf 0(%0)" : : "r" (post_edge_vfblockaddr));
       k += consumed;
     }
   }
   if (ALPHA > 1.01 || ALPHA < 0.99)
     scalex_h (C, ALPHA, M*N);
   asm volatile ("fence");
}

void gemm_nn_hwacha_h_eff (int M, int N, int K, float ALPHA,
                           int16_t* A, int lda,
                           int16_t* B, int ldb,
                           int16_t* C, int ldc)
{
  asm volatile ("vsetcfg %0" : : "r" (VCFG(0, 0, 9, 1)));
  int vlen_result;
  void * vpset_vfblockaddr;
  asm volatile ("la %0, hgemm_eff_vpset" : "=r" (vpset_vfblockaddr));
  asm volatile ("vf 0(%0)" : : "r" (vpset_vfblockaddr));


  asm volatile ("la t0, hgemm_eff_pre" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_top" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_0" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_1" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_2" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_3" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_4" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_5" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_6" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_7" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_post" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_pre_edge" : : : "t0");
  asm volatile ("lw t1, 0(t0)");
  asm volatile ("la t0, hgemm_eff_post_edge" : : : "t0");
  asm volatile ("lw t1, 0(t0)");

  int i;
  for (i = 0; i + 8 <= M; i += 8) {
    for (int k = 0; k < N; ) {
      int consumed = setvlen(N - k);
      //C rows 1, 2, 3, 4, 5, 6, 7, 8
      asm volatile ("vmca va0, %0" : : "r" (&C[(i+0)*ldc + k]));
      asm volatile ("vmca va1, %0" : : "r" (&C[(i+1)*ldc + k]));
      asm volatile ("vmca va2, %0" : : "r" (&C[(i+2)*ldc + k]));
      asm volatile ("vmca va3, %0" : : "r" (&C[(i+3)*ldc + k]));
      asm volatile ("vmca va4, %0" : : "r" (&C[(i+4)*ldc + k]));
      asm volatile ("vmca va5, %0" : : "r" (&C[(i+5)*ldc + k]));
      asm volatile ("vmca va6, %0" : : "r" (&C[(i+6)*ldc + k]));
      asm volatile ("vmca va7, %0" : : "r" (&C[(i+7)*ldc + k]));
      asm volatile ("la t0, hgemm_eff_pre" : : : "t0");
      asm volatile ("vf 0(t0)");
      for (int j = 0; j < K; j++) {
        asm volatile ("vmca va8, %0" : : "r" (&B[j*ldb+k]));
        asm volatile ("la t0, hgemm_eff_top" : : : "t0");
        asm volatile ("vf 0(t0)");

        if (A[j+(i*0)*lda] != 0) {
          asm volatile ("vmcs vs1, %0" : : "r" (A[j+(i*0)*lda]));
          asm volatile ("la t0, hgemm_eff_0" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
        if (A[j+(i*1)*lda] != 0) {
          asm volatile ("vmcs vs2, %0" : : "r" (A[j+(i*1)*lda]));
          asm volatile ("la t0, hgemm_eff_1" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
        if (A[j+(i*2)*lda] != 0) {
          asm volatile ("vmcs vs3, %0" : : "r" (A[j+(i*2)*lda]));
          asm volatile ("la t0, hgemm_eff_2" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
        if (A[j+(i*3)*lda] != 0) {
          asm volatile ("vmcs vs4, %0" : : "r" (A[j+(i*3)*lda]));
          asm volatile ("la t0, hgemm_eff_3" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
        if (A[j+(i*4)*lda] != 0) {
          asm volatile ("vmcs vs5, %0" : : "r" (A[j+(i*4)*lda]));
          asm volatile ("la t0, hgemm_eff_4" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
        if (A[j+(i*5)*lda] != 0) {
          asm volatile ("vmcs vs6, %0" : : "r" (A[j+(i*5)*lda]));
          asm volatile ("la t0, hgemm_eff_5" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
        if (A[j+(i*6)*lda] != 0) {
          asm volatile ("vmcs vs7, %0" : : "r" (A[j+(i*6)*lda]));
          asm volatile ("la t0, hgemm_eff_6" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
        if (A[j+(i*7)*lda] != 0) {
          asm volatile ("vmcs vs8, %0" : : "r" (A[j+(i*7)*lda]));
          asm volatile ("la t0, hgemm_eff_7" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
      }
      asm volatile ("la t0, hgemm_eff_post" : : : "t0");
      asm volatile ("vf 0(t0)");
      k += consumed;
    }
  }
  for ( ; i < M; i++)
    {
      for (int k = 0; k < N; ) {
      int consumed = setvlen(N - k);
      //C rows 1, 2, 3, 4, 5, 6, 7, 8
      asm volatile ("vmca va0, %0" : : "r" (&C[(i+0)*ldc + k]));
      asm volatile ("la t0, hgemm_eff_pre_edge" : : : "t0");
      asm volatile ("vf 0(t0)");
      for (int j = 0; j < K; j++) {
        if (A[j+(i*0)*lda] != 0) {
          asm volatile ("vmcs vs1, %0" : : "r" (A[j+(i*0)*lda]));
          asm volatile ("la t0, hgemm_eff_0" : : : "t0");
          asm volatile ("vf 0(t0)");
        }
      }
      asm volatile ("la t0, hgemm_eff_post_edge" : : : "t0");
      asm volatile ("vf 0(t0)");
      k += consumed;
      }
    }
  if (ALPHA > 1.01 || ALPHA < 0.99)
    scalex_h(C, ALPHA, M*N);
  asm volatile ("fence");


}

void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  gemm_nn_hwacha(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  return;
}


void gemm_nn_h(int M, int N, int K, float ALPHA,
        int16_t *A, int lda,
        int16_t *B, int ldb,
        int16_t *C, int ldc)
{
  gemm_nn_hwacha_h(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  return;
}

void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
  if (BETA > 1.00001 || BETA < 0.99999)
    {
      scalex(C, BETA, M*N);
    }

    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm_cpu_h(int TA, int TB, int M, int N, int K, float ALPHA,
        int16_t *A, int lda,
        int16_t *B, int ldb,
        float BETA,
        int16_t *C, int ldc)
{

  if (BETA > 1.00001 || BETA < 0.99999)
    {
      scalex_h (C, BETA, M*N);
    }
  if(!TA && !TB)
    gemm_nn_h(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);

}

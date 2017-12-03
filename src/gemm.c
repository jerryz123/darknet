#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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
                       : "r" (ALPHA*A[j+(i+0)*lda+0]), "r" (ALPHA*A[j+(i+0)*lda+1]), "r" (ALPHA*A[j+(i+0)*lda+2]), "r" (ALPHA*A[j+(i+0)*lda+3]),
                         "r" (ALPHA*A[j+(i+1)*lda+0]), "r" (ALPHA*A[j+(i+1)*lda+1]), "r" (ALPHA*A[j+(i+1)*lda+2]), "r" (ALPHA*A[j+(i+1)*lda+3]),
                         "r" (ALPHA*A[j+(i+2)*lda+0]), "r" (ALPHA*A[j+(i+2)*lda+1]), "r" (ALPHA*A[j+(i+2)*lda+2]), "r" (ALPHA*A[j+(i+2)*lda+3]),
                         "r" (ALPHA*A[j+(i+3)*lda+0]), "r" (ALPHA*A[j+(i+3)*lda+1]), "r" (ALPHA*A[j+(i+3)*lda+2]), "r" (ALPHA*A[j+(i+3)*lda+3])
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
                       : "r" (ALPHA*A[j+(i+0)*lda+0]),
                         "r" (ALPHA*A[j+(i+1)*lda+0]),
                         "r" (ALPHA*A[j+(i+2)*lda+0]),
                         "r" (ALPHA*A[j+(i+3)*lda+0])
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
         asm volatile ("vmcs vs1, %0" : : "r" (ALPHA*A[j+(i+0)*lda+0]));
         asm volatile ("vf 0(%0)" : : "r" (main_edge1_vfblockaddr));
       }
       asm volatile ("vf 0(%0)" : : "r" (post_edge_vfblockaddr));
       k += consumed;
     }
   }
   asm volatile ("fence");
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
  gemm_nn_hwacha(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  return;
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
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
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
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

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
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
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif


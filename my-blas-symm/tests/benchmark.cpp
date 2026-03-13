#include "../src/symm.h"
#include <cblas.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

using hrc = std::chrono::high_resolution_clock;

int main() {
    const int M = 1024;
    const int N = 1024;
    const int lda = M, ldb = M, ldc = M; 

    float *A = new float[M * lda];
    float *B = new float[M * ldb];
    float *C_my = new float[M * ldc];
    float *C_ref = new float[M * ldc];

    srand(42);
    for (int i = 0; i < M * lda; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < M * ldb; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

    std::vector<int> threads = {1, 2, 4, 8, 16};

    std::cout << "Тест my_ssymm  (M=" << M << ", N=" << N << ", side=L, uplo=U)\n\n";

    for (int t : threads) {
        openblas_set_num_threads(t);

        auto start = hrc::now();
        for (int run = 0; run < 10; ++run) {
            cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
                        M, N, 1.0f, A, lda, B, ldb, 0.0f, C_ref, ldc);
        }
        double time_ref = std::chrono::duration<double>(hrc::now() - start).count() / 10.0;
        double gflops_ref = 2.0 * M * N * M / (time_ref * 1e9);  

        start = hrc::now();
        for (int run = 0; run < 10; ++run) {
            std::memcpy(C_my, C_ref, sizeof(float) * M * ldc); 
            my_ssymm('L', 'U', M, N, 1.0f, A, lda, B, ldb, 0.0f, C_my, ldc);
        }
        double time_my = std::chrono::duration<double>(hrc::now() - start).count() / 10.0;
        double gflops_my = 2.0 * M * N * M / (time_my * 1e9);

        double percent = (gflops_my / gflops_ref) * 100.0;

        std::cout << "Потоков: " << t
                  << " | OpenBLAS: " << gflops_ref << " GFLOPS (" << time_ref << " с)"
                  << " | Моя: " << gflops_my << " GFLOPS (" << time_my << " с)"
                  << " | " << percent << " %\n";
    }

    delete[] A; delete[] B; delete[] C_my; delete[] C_ref;
    return 0;
}
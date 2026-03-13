#include "symm.h"
#include <algorithm>
#include <cstring>
#include <cctype> 

const int BLOCK_M = 128;
const int BLOCK_N = 128;
const int BLOCK_K = 64;

static inline float get_symm_element(const float* A, int lda, int i, int j, char uplo) {
    if (std::toupper(uplo) == 'U') {
        return (i <= j) ? A[j * lda + i] : A[i * lda + j];
    } else {
        return (i >= j) ? A[j * lda + i] : A[i * lda + j];
    }
}

static inline double get_symm_element(const double* A, int lda, int i, int j, char uplo) {
    if (std::toupper(uplo) == 'U') {
        return (i <= j) ? A[j * lda + i] : A[i * lda + j];
    } else {
        return (i >= j) ? A[j * lda + i] : A[i * lda + j];
    }
}

void my_ssymm(char side, char uplo,
              int m, int n,
              float alpha, const float* A, int lda,
              const float* B, int ldb,
              float beta, float* C, int ldc) {

    char side_up = std::toupper(side);
    char uplo_up = std::toupper(uplo);

    if (beta != 1.0f) {
        if (beta == 0.0f) {
            for (int i = 0; i < m; ++i)
                std::fill(C + i * ldc, C + i * ldc + n, 0.0f);
        } else {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C[i * ldc + j] *= beta;
        }
    }

    if (side_up == 'L') { 
        for (int kk = 0; kk < m; kk += BLOCK_K) {
            int kb = std::min(BLOCK_K, m - kk);
            for (int ii = 0; ii < m; ii += BLOCK_M) {
                int mb = std::min(BLOCK_M, m - ii);
                for (int jj = 0; jj < n; jj += BLOCK_N) {
                    int nb = std::min(BLOCK_N, n - jj);

                    for (int i = ii; i < ii + mb; ++i) {
                        for (int j = jj; j < jj + nb; ++j) {
                            float sum = 0.0f;
                            for (int p = kk; p < kk + kb; ++p) {
                                sum += get_symm_element(A, lda, i, p, uplo_up) * B[p * ldb + j];
                            }
                            C[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    } else { 
        for (int kk = 0; kk < n; kk += BLOCK_K) {
            int kb = std::min(BLOCK_K, n - kk);
            for (int ii = 0; ii < m; ii += BLOCK_M) {
                int mb = std::min(BLOCK_M, m - ii);
                for (int jj = 0; jj < n; jj += BLOCK_N) {
                    int nb = std::min(BLOCK_N, n - jj);

                    for (int i = ii; i < ii + mb; ++i) {
                        for (int j = jj; j < jj + nb; ++j) {
                            float sum = 0.0f;
                            for (int p = kk; p < kk + kb; ++p) {
                                sum += B[i * ldb + p] * get_symm_element(A, lda, p, j, uplo_up);
                            }
                            C[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }
}

void my_dsymm(char side, char uplo,
              int m, int n,
              double alpha, const double* A, int lda,
              const double* B, int ldb,
              double beta, double* C, int ldc) {

    char side_up = std::toupper(side);
    char uplo_up = std::toupper(uplo);

    if (beta != 1.0) {
        if (beta == 0.0) {
            for (int i = 0; i < m; ++i)
                std::fill(C + i * ldc, C + i * ldc + n, 0.0);
        } else {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C[i * ldc + j] *= beta;
        }
    }

    if (side_up == 'L') {
        for (int kk = 0; kk < m; kk += BLOCK_K) {
            int kb = std::min(BLOCK_K, m - kk);
            for (int ii = 0; ii < m; ii += BLOCK_M) {
                int mb = std::min(BLOCK_M, m - ii);
                for (int jj = 0; jj < n; jj += BLOCK_N) {
                    int nb = std::min(BLOCK_N, n - jj);

                    for (int i = ii; i < ii + mb; ++i) {
                        for (int j = jj; j < jj + nb; ++j) {
                            double sum = 0.0;
                            for (int p = kk; p < kk + kb; ++p) {
                                sum += get_symm_element(A, lda, i, p, uplo_up) * B[p * ldb + j];
                            }
                            C[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    } else {
        for (int kk = 0; kk < n; kk += BLOCK_K) {
            int kb = std::min(BLOCK_K, n - kk);
            for (int ii = 0; ii < m; ii += BLOCK_M) {
                int mb = std::min(BLOCK_M, m - ii);
                for (int jj = 0; jj < n; jj += BLOCK_N) {
                    int nb = std::min(BLOCK_N, n - jj);

                    for (int i = ii; i < ii + mb; ++i) {
                        for (int j = jj; j < jj + nb; ++j) {
                            double sum = 0.0;
                            for (int p = kk; p < kk + kb; ++p) {
                                sum += B[i * ldb + p] * get_symm_element(A, lda, p, j, uplo_up);
                            }
                            C[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }
}
#ifndef SYMM_H
#define SYMM_H

void my_ssymm(char side, char uplo,
              int m, int n,
              float alpha, const float* A, int lda,
              const float* B, int ldb,
              float beta, float* C, int ldc);

void my_dsymm(char side, char uplo,
              int m, int n,
              double alpha, const double* A, int lda,
              const double* B, int ldb,
              double beta, double* C, int ldc);

#endif
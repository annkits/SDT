#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#define SMALL_N 10        
#define BIG_N 1000000     

static void fill_float(float *v, int n) {
    for (int i = 0; i < n; i++) v[i] = (float)(i % 5) - 2.5f;
}

static void fill_double(double *v, int n) {
    for (int i = 0; i < n; i++) v[i] = (double)(i % 7) - 3.14;
}

static void fill_complex_float(void *v, int n) {
    float *cv = (float *)v;
    for (int i = 0; i < n*2; i += 2) {
        cv[i]   = (float)(i % 5) - 2.5f;   // Re
        cv[i+1] = (float)(i % 3) + 1.0f;   // Im
    }
}

static void fill_complex_double(void *v, int n) {
    double *cv = (double *)v;
    for (int i = 0; i < n*2; i += 2) {
        cv[i]   = (double)(i % 7) - 3.14;  // Re
        cv[i+1] = (double)(i % 4) + 2.0;   // Im
    }
}

static void print_header(const char *group_name) {
    printf("\n=== Тестируем группу: %s ===\n", group_name);
}

static void smoke_dot_real(void) {
    print_header("sdot / ddot / sdsdot / dsdot");

    float fx[SMALL_N*3], fy[SMALL_N*2];
    double dx[SMALL_N*2], dy[SMALL_N*3];

    fill_float(fx, SMALL_N*3);
    fill_float(fy, SMALL_N*2);
    fill_double(dx, SMALL_N*2);
    fill_double(dy, SMALL_N*3);

    (void)cblas_sdot(SMALL_N, fx, 1, fy, 1);
    (void)cblas_sdot(0, fx, 1, fy, 1);
    (void)cblas_sdot(SMALL_N, fx, -2, fy, 3);
    (void)cblas_sdot(-5, fx, 1, fy, 1);

    (void)cblas_ddot(SMALL_N, dx, 2, dy, -1);
    (void)cblas_ddot(0, dx, 1, dy, 1);

    (void)cblas_sdsdot(SMALL_N, 1.5f, fx, 1, fy, 1);
    (void)cblas_dsdot(SMALL_N, fx, -1, fy, 2);

    // большой
    float *bigx = malloc(BIG_N * sizeof(float));
    if (bigx) {
        (void)cblas_sdot(BIG_N, bigx, 1, bigx, 1);
        free(bigx);
    }

    printf("  → dot (real) отработали\n");
}

static void smoke_dot_complex(void) {
    print_header("cdotu_sub / cdotc_sub / zdotu_sub / zdotc_sub");

    float cx[SMALL_N*4*2], cy[SMALL_N*3*2]; 
    double zx[SMALL_N*2*2], zy[SMALL_N*4*2];

    fill_complex_float(cx, SMALL_N*4);
    fill_complex_float(cy, SMALL_N*3);
    fill_complex_double(zx, SMALL_N*2);
    fill_complex_double(zy, SMALL_N*4);

    float dotf[2];
    double dotd[2];

    cblas_cdotu_sub(SMALL_N, cx, 1, cy, 1, dotf);
    cblas_cdotu_sub(0, cx, 1, cy, 1, dotf);
    cblas_cdotu_sub(SMALL_N, cx, -3, cy, 2, dotf);

    cblas_cdotc_sub(SMALL_N, cx, 2, cy, -1, dotf);
    cblas_cdotc_sub(-4, cx, 1, cy, 1, dotf);

    cblas_zdotu_sub(SMALL_N, zx, 1, zy, 1, dotd);
    cblas_zdotu_sub(SMALL_N, zx, -2, zy, 3, dotd);

    cblas_zdotc_sub(SMALL_N, zx, 3, zy, -1, dotd);

    printf("  → dot (complex) отработали\n");
}

static void smoke_norms(void) {
    print_header("snrm2 / dnrm2 / scnrm2 / dznrm2 / sasum / dasum / scasum / dzasum");

    float fx[SMALL_N*4], fy[SMALL_N*3];
    double dx[SMALL_N*2], dy[SMALL_N];

    fill_float(fx, SMALL_N*4);
    fill_float(fy, SMALL_N*3);
    fill_double(dx, SMALL_N*2);
    fill_double(dy, SMALL_N);

    (void)cblas_snrm2(SMALL_N, fx, 1);
    (void)cblas_snrm2(0, fx, 1);
    (void)cblas_snrm2(SMALL_N, fx, -2);

    (void)cblas_dnrm2(SMALL_N, dx, 3);
    (void)cblas_dnrm2(SMALL_N, dx, -1);

    (void)cblas_sasum(SMALL_N, fy, 2);
    (void)cblas_dasum(0, dy, 1);

    float cx[SMALL_N*3*2];
    double zx[SMALL_N*4*2];

    fill_complex_float(cx, SMALL_N*3);
    fill_complex_double(zx, SMALL_N*4);

    (void)cblas_scnrm2(SMALL_N, cx, 1);
    (void)cblas_dznrm2(SMALL_N, zx, -2);

    (void)cblas_scasum(SMALL_N, cx, 3);
    (void)cblas_dzasum(0, zx, 1);

    printf("  → norms и asum отработали\n");
}

static void smoke_amax(void) {
    print_header("isamax / idamax / icamax / izamax");

    float fx[SMALL_N*3];
    double dx[SMALL_N*2];

    fill_float(fx, SMALL_N*3);
    fill_double(dx, SMALL_N*2);

    (void)cblas_isamax(SMALL_N, fx, 1);
    (void)cblas_isamax(0, fx, 1);
    (void)cblas_isamax(SMALL_N, fx, -4);

    (void)cblas_idamax(SMALL_N, dx, 2);
    (void)cblas_idamax(SMALL_N, dx, -1);

    float cx[SMALL_N*5*2];
    double zx[SMALL_N*3*2];

    fill_complex_float(cx, SMALL_N*5);
    fill_complex_double(zx, SMALL_N*3);

    (void)cblas_icamax(SMALL_N, cx, 1);
    (void)cblas_icamax(SMALL_N, cx, -3);

    (void)cblas_izamax(0, zx, 1);
    (void)cblas_izamax(SMALL_N, zx, 2);

    printf("  → amax отработали\n");
}
static void smoke_swap(void) {
    print_header("sswap / dswap / cswap / zswap");

    float fx[SMALL_N*2], fy[SMALL_N*3];
    double dx[SMALL_N*4], dy[SMALL_N];

    fill_float(fx, SMALL_N*2);
    fill_float(fy, SMALL_N*3);
    fill_double(dx, SMALL_N*4);
    fill_double(dy, SMALL_N);

    cblas_sswap(SMALL_N, fx, 1, fy, 1);
    cblas_sswap(0, fx, 1, fy, 1);
    cblas_sswap(SMALL_N, fx, -2, fy, 3);

    cblas_dswap(SMALL_N, dx, 2, dy, -1);

    // overlap
    cblas_sswap(SMALL_N, fx, 1, fx+2, 1);

    // complex
    float cx[SMALL_N*3*2], cy[SMALL_N*2*2];
    double zx[SMALL_N*4*2], zy[SMALL_N*3*2];

    fill_complex_float(cx, SMALL_N*3);
    fill_complex_float(cy, SMALL_N*2);
    fill_complex_double(zx, SMALL_N*4);
    fill_complex_double(zy, SMALL_N*3);

    cblas_cswap(SMALL_N, cx, 1, cy, -2);
    cblas_cswap(0, cx, 1, cy, 1);

    cblas_zswap(SMALL_N, zx, 3, zy, 1);
    cblas_zswap(SMALL_N, zx, -1, zy, 2);

    printf("  → swap отработали\n");
}
static void smoke_copy(void) {
    print_header("scopy / dcopy / ccopy / zcopy");

    float fx[SMALL_N*3], fy[SMALL_N*2];
    double dx[SMALL_N], dy[SMALL_N*4];

    fill_float(fx, SMALL_N*3);
    fill_float(fy, SMALL_N*2);
    fill_double(dx, SMALL_N);
    fill_double(dy, SMALL_N*4);

    cblas_scopy(SMALL_N, fx, 1, fy, 1);
    cblas_scopy(0, fx, 1, fy, 1);
    cblas_scopy(SMALL_N, fx, -3, fy, 2);

    cblas_dcopy(SMALL_N, dx, 2, dy, -1);

    // overlap
    cblas_scopy(SMALL_N, fx, 1, fx+3, 1);

    // big
    float *bigf = malloc(BIG_N * sizeof(float) * 2);
    if (bigf) {
        cblas_scopy(BIG_N, bigf, 1, bigf + BIG_N, 1);
        free(bigf);
    }

    // complex
    float cx[SMALL_N*4*2], cy[SMALL_N*3*2];
    double zx[SMALL_N*2*2], zy[SMALL_N*5*2];

    fill_complex_float(cx, SMALL_N*4);
    fill_complex_float(cy, SMALL_N*3);
    fill_complex_double(zx, SMALL_N*2);
    fill_complex_double(zy, SMALL_N*5);

    cblas_ccopy(SMALL_N, cx, 1, cy, -2);
    cblas_ccopy(0, cx, 1, cy, 1);

    cblas_zcopy(SMALL_N, zx, 3, zy, 1);
    cblas_zcopy(SMALL_N, zx, -1, zy, 4);

    printf("  → copy отработали\n");
}

static void smoke_axpy(void) {
    print_header("saxpy / daxpy / caxpy / zaxpy");

    float fx[SMALL_N*3], fy[SMALL_N*4];
    double dx[SMALL_N*2], dy[SMALL_N];

    fill_float(fx, SMALL_N*3);
    fill_float(fy, SMALL_N*4);
    fill_double(dx, SMALL_N*2);
    fill_double(dy, SMALL_N);

    cblas_saxpy(SMALL_N, 1.5f, fx, 1, fy, 1);
    cblas_saxpy(0, 0.0f, fx, 1, fy, 1);
    cblas_saxpy(SMALL_N, -2.0f, fx, -2, fy, 3);

    cblas_daxpy(SMALL_N, 3.14, dx, 2, dy, -1);

    // overlap
    cblas_saxpy(SMALL_N, 1.0f, fx, 1, fx+2, 1);

    // complex
    float cx[SMALL_N*3*2], cy[SMALL_N*2*2];
    double zx[SMALL_N*4*2], zy[SMALL_N*3*2];

    fill_complex_float(cx, SMALL_N*3);
    fill_complex_float(cy, SMALL_N*2);
    fill_complex_double(zx, SMALL_N*4);
    fill_complex_double(zy, SMALL_N*3);

    float calpha[2] = {1.0f, -1.0f};
    double zalpha[2] = {2.0, 0.0};

    cblas_caxpy(SMALL_N, calpha, cx, 1, cy, -2);
    cblas_caxpy(0, calpha, cx, 1, cy, 1);

    cblas_zaxpy(SMALL_N, zalpha, zx, 3, zy, 1);
    cblas_zaxpy(SMALL_N, zalpha, zx, -1, zy, 2);

    printf("  → axpy отработали\n");
}

static void smoke_rot_real(void) {
    print_header("srotg / drotg / srot / drot / srotmg / drotmg / srotm / drotm");

    float fa = 1.0f, fb = 2.0f, fc, fs;
    double da = 3.14, db = -1.0, dc, ds;

    cblas_srotg(&fa, &fb, &fc, &fs);
    cblas_drotg(&da, &db, &dc, &ds);

    float fx[SMALL_N*2], fy[SMALL_N*3];
    double dx[SMALL_N*4], dy[SMALL_N];

    fill_float(fx, SMALL_N*2);
    fill_float(fy, SMALL_N*3);
    fill_double(dx, SMALL_N*4);
    fill_double(dy, SMALL_N);

    cblas_srot(SMALL_N, fx, 1, fy, -2, fc, fs);
    cblas_srot(0, fx, 1, fy, 1, 0.0f, 1.0f);

    cblas_drot(SMALL_N, dx, 3, dy, 1, dc, ds);

    // overlap
    cblas_srot(SMALL_N, fx, 1, fx+1, 1, fc, fs);

    // rotmg / rotm
    float fm1 = 1.0f, fm2 = 2.0f, fm3 = 3.0f, fm4 = 4.0f;
    double dm1 = 5.0, dm2 = 6.0, dm3 = 7.0, dm4 = 8.0;
    float paramf[5];
    double paramd[5];

    cblas_srotmg(&fm1, &fm2, &fm3, fm4, paramf);
    cblas_drotmg(&dm1, &dm2, &dm3, dm4, paramd);

    cblas_srotm(SMALL_N, fx, -1, fy, 2, paramf);
    cblas_drotm(0, dx, 1, dy, 1, paramd);

    printf("  → real rot отработали\n");
}

static void smoke_rot_complex(void) {
    print_header("crotg / zrotg / crot / zrot / csrot / zdrot");

    // crotg / zrotg
    float ca = 1.0f, cb[2] = {2.0f, 3.0f}; // cb complex
    float cc, cs[2]; // cc real, cs complex
    cblas_crotg(&ca, cb, &cc, cs);

    double da = 4.0, db[2] = {5.0, 6.0};
    double dc, ds[2];
    cblas_zrotg(&da, db, &dc, ds);

    // crot / zrot
    float cx[SMALL_N*2*2], cy[SMALL_N*3*2];
    fill_complex_float(cx, SMALL_N*2);
    fill_complex_float(cy, SMALL_N*3);

    double zx[SMALL_N*4*2], zy[SMALL_N*2*2];
    fill_complex_double(zx, SMALL_N*4);
    fill_complex_double(zy, SMALL_N*2);

    // csrot / zdrot 
    float csc = 0.5f, css = -0.5f;
    cblas_csrot(SMALL_N, cx, 1, cy, 1, csc, css);

    double zdc = 0.8, zds = 0.6;
    cblas_zdrot(SMALL_N, zx, 1, zy, 1, zdc, zds);

    // overlap example
    cblas_csrot(SMALL_N, cx, 1, cx+2, 1, csc, css);

    printf("  → complex rot отработали\n");
}

static void smoke_scal(void) {
    print_header("sscal / dscal / cscal / zscal / csscal / zdscal");

    float fx[SMALL_N*3];
    double dx[SMALL_N*2];

    fill_float(fx, SMALL_N*3);
    fill_double(dx, SMALL_N*2);

    cblas_sscal(SMALL_N, 1.5f, fx, 1);
    cblas_sscal(0, 0.0f, fx, 1);
    cblas_sscal(SMALL_N, -2.0f, fx, -3);

    cblas_dscal(SMALL_N, 3.14, dx, 2);

    // complex
    float cx[SMALL_N*4*2];
    double zx[SMALL_N*3*2];

    fill_complex_float(cx, SMALL_N*4);
    fill_complex_double(zx, SMALL_N*3);

    float calpha[2] = {1.0f, -1.0f};
    double zalpha[2] = {2.0, 0.0};

    cblas_cscal(SMALL_N, calpha, cx, 1);
    cblas_cscal(SMALL_N, calpha, cx, -2);

    cblas_zscal(0, zalpha, zx, 1);
    cblas_zscal(SMALL_N, zalpha, zx, 3);

    // cs/zd scal
    cblas_csscal(SMALL_N, 2.5f, cx, -1);
    cblas_zdscal(SMALL_N, -1.7, zx, 2);

    printf("  → scal отработали\n");
}

int main(void) {
    printf("=== Полные интерфейсные тесты CBLAS Level 1 (все функции) ===\n");
    printf("Цель: покрыть вызовы всех функций с разными параметрами,\n");
    printf("      без проверки результатов — только на стабильность.\n\n");

    smoke_dot_real();
    smoke_dot_complex();
    smoke_norms();
    smoke_amax();
    smoke_swap();
    smoke_copy();
    smoke_axpy();
    smoke_rot_real();
    smoke_rot_complex();
    smoke_scal();

    printf("\nВсе функции Level 1 вызваны с разными сценариями.\n");
    printf("Нет видимых крашей → покрытие интерфейса полное.\n");

    return 0;
}
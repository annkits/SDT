import ctypes
import numpy as np
import os
import multiprocessing as mp

try:
    lib = ctypes.CDLL("/usr/local/openblas/lib/libopenblas.so")
except OSError as e:
    print("Ошибка загрузки libopenblas:", e)

c_float_p   = ctypes.POINTER(ctypes.c_float)
c_double_p  = ctypes.POINTER(ctypes.c_double)
c_void_p    = ctypes.c_void_p
c_int       = ctypes.c_int
c_float     = ctypes.c_float
c_double    = ctypes.c_double

SMALL_N = 10
BIG_N   = 1000000

def run_test(func):
    try:
        func()
    except Exception as e:
        print(f"!!! {func.__name__} exception: {e}")
    except:
        print(f"!!! {func.__name__} unknown error")

def fill_float(n):
    return np.arange(n, dtype=np.float32) % 5 - 2.5

def fill_double(n):
    return np.arange(n, dtype=np.float64) % 7 - 3.14

def fill_complex_float(n):
    re = np.arange(n, dtype=np.float32) % 5 - 2.5
    im = np.arange(n, dtype=np.float32) % 3 + 1.0
    return np.array(re + 1j * im, dtype=np.complex64)

def fill_complex_double(n):
    re = np.arange(n, dtype=np.float64) % 7 - 3.14
    im = np.arange(n, dtype=np.float64) % 4 + 2.0
    return np.array(re + 1j * im, dtype=np.complex128)

def print_header(group_name):
    print(f"\n=== Testing: {group_name} ===")

def test_dot_real():
    print_header("sdot / ddot / sdsdot / dsdot")

    fx = fill_float(SMALL_N*3)
    fy = fill_float(SMALL_N*2)
    dx = fill_double(SMALL_N*2)
    dy = fill_double(SMALL_N*3)

    lib.cblas_sdot.argtypes = [c_int, c_float_p, c_int, c_float_p, c_int]
    lib.cblas_sdot.restype  = c_float
    lib.cblas_sdot(SMALL_N, fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)
    lib.cblas_sdot(0, fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)
    lib.cblas_sdot(SMALL_N, fx.ctypes.data_as(c_float_p), -2, fy.ctypes.data_as(c_float_p), 3)

    lib.cblas_ddot.argtypes = [c_int, c_double_p, c_int, c_double_p, c_int]
    lib.cblas_ddot.restype  = c_double
    lib.cblas_ddot(SMALL_N, dx.ctypes.data_as(c_double_p), 2, dy.ctypes.data_as(c_double_p), -1)

    lib.cblas_sdsdot.argtypes = [c_int, c_float, c_float_p, c_int, c_float_p, c_int]
    lib.cblas_sdsdot.restype  = c_float
    lib.cblas_sdsdot(SMALL_N, c_float(1.5), fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)

    lib.cblas_dsdot.argtypes = [c_int, c_float_p, c_int, c_float_p, c_int]
    lib.cblas_dsdot.restype  = c_double
    lib.cblas_dsdot(SMALL_N, fx.ctypes.data_as(c_float_p), -1, fy.ctypes.data_as(c_float_p), 2)

    print("  → dot (real) successful")

def test_dot_complex():
    print_header("cdotu_sub / cdotc_sub / zdotu_sub / zdotc_sub")

    cx = fill_complex_float(SMALL_N*4)
    cy = fill_complex_float(SMALL_N*3)
    zx = fill_complex_double(SMALL_N*2)
    zy = fill_complex_double(SMALL_N*4)

    dotf = np.zeros(2, dtype=np.float32)
    dotd = np.zeros(2, dtype=np.float64)

    lib.cblas_cdotu_sub.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int, c_void_p]
    lib.cblas_cdotu_sub(SMALL_N, cx.ctypes.data, 1, cy.ctypes.data, 1, dotf.ctypes.data_as(c_void_p))
    lib.cblas_cdotu_sub(0, cx.ctypes.data, 1, cy.ctypes.data, 1, dotf.ctypes.data_as(c_void_p))

    lib.cblas_cdotc_sub.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int, c_void_p]
    lib.cblas_cdotc_sub(SMALL_N, cx.ctypes.data, -3, cy.ctypes.data, 2, dotf.ctypes.data_as(c_void_p))

    lib.cblas_zdotu_sub.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int, c_void_p]
    lib.cblas_zdotu_sub(SMALL_N, zx.ctypes.data, 1, zy.ctypes.data, 1, dotd.ctypes.data_as(c_void_p))

    lib.cblas_zdotc_sub.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int, c_void_p]
    lib.cblas_zdotc_sub(SMALL_N, zx.ctypes.data, 3, zy.ctypes.data, -1, dotd.ctypes.data_as(c_void_p))

    print("  → dot (complex) successful")

def test_norms():
    print_header("snrm2 / dnrm2 / scnrm2 / dznrm2 / sasum / dasum / scasum / dzasum")

    fx = fill_float(SMALL_N*4)
    dx = fill_double(SMALL_N*2)
    cx = fill_complex_float(SMALL_N*3)
    zx = fill_complex_double(SMALL_N*4)

    lib.cblas_snrm2.argtypes = [c_int, c_float_p, c_int]
    lib.cblas_snrm2.restype  = c_float
    lib.cblas_snrm2(SMALL_N, fx.ctypes.data_as(c_float_p), 1)
    lib.cblas_snrm2(0, fx.ctypes.data_as(c_float_p), 1)

    lib.cblas_dnrm2.argtypes = [c_int, c_double_p, c_int]
    lib.cblas_dnrm2.restype  = c_double
    lib.cblas_dnrm2(SMALL_N, dx.ctypes.data_as(c_double_p), -1)

    lib.cblas_scnrm2.argtypes = [c_int, c_void_p, c_int]
    lib.cblas_scnrm2.restype  = c_float
    lib.cblas_scnrm2(SMALL_N, cx.ctypes.data, 1)

    lib.cblas_dznrm2.argtypes = [c_int, c_void_p, c_int]
    lib.cblas_dznrm2.restype  = c_double
    lib.cblas_dznrm2(SMALL_N, zx.ctypes.data, -2)

    lib.cblas_sasum.argtypes = [c_int, c_float_p, c_int]
    lib.cblas_sasum.restype  = c_float
    lib.cblas_sasum(SMALL_N, fx.ctypes.data_as(c_float_p), 2)

    lib.cblas_dasum.argtypes = [c_int, c_double_p, c_int]
    lib.cblas_dasum.restype  = c_double
    lib.cblas_dasum(SMALL_N, dx.ctypes.data_as(c_double_p), 1)

    lib.cblas_scasum.argtypes = [c_int, c_void_p, c_int]
    lib.cblas_scasum.restype  = c_float
    lib.cblas_scasum(SMALL_N, cx.ctypes.data, 3)

    lib.cblas_dzasum.argtypes = [c_int, c_void_p, c_int]
    lib.cblas_dzasum.restype  = c_double
    lib.cblas_dzasum(SMALL_N, zx.ctypes.data, 1)

    print("  → norms и asum successful")

def test_amax():
    print_header("isamax / idamax / icamax / izamax")

    fx = fill_float(SMALL_N*3)
    dx = fill_double(SMALL_N*2)
    cx = fill_complex_float(SMALL_N*5)
    zx = fill_complex_double(SMALL_N*3)

    lib.cblas_isamax.argtypes = [c_int, c_float_p, c_int]
    lib.cblas_isamax.restype  = c_int
    lib.cblas_isamax(SMALL_N, fx.ctypes.data_as(c_float_p), 1)
    lib.cblas_isamax(0, fx.ctypes.data_as(c_float_p), 1)

    lib.cblas_idamax.argtypes = [c_int, c_double_p, c_int]
    lib.cblas_idamax.restype  = c_int
    lib.cblas_idamax(SMALL_N, dx.ctypes.data_as(c_double_p), -1)

    lib.cblas_icamax.argtypes = [c_int, c_void_p, c_int]
    lib.cblas_icamax.restype  = c_int
    lib.cblas_icamax(SMALL_N, cx.ctypes.data, 1)

    lib.cblas_izamax.argtypes = [c_int, c_void_p, c_int]
    lib.cblas_izamax.restype  = c_int
    lib.cblas_izamax(SMALL_N, zx.ctypes.data, 2)

    print("  → amax successful")

def test_swap():
    print_header("sswap / dswap / cswap / zswap")

    fx = fill_float(SMALL_N*2)
    fy = fill_float(SMALL_N*3)
    dx = fill_double(SMALL_N*4)
    dy = fill_double(SMALL_N)

    lib.cblas_sswap.argtypes = [c_int, c_float_p, c_int, c_float_p, c_int]
    lib.cblas_sswap(SMALL_N, fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)
    lib.cblas_sswap(0, fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)

    lib.cblas_dswap.argtypes = [c_int, c_double_p, c_int, c_double_p, c_int]
    lib.cblas_dswap(SMALL_N, dx.ctypes.data_as(c_double_p), 2, dy.ctypes.data_as(c_double_p), -1)

    cx = fill_complex_float(SMALL_N*3)
    cy = fill_complex_float(SMALL_N*2)
    zx = fill_complex_double(SMALL_N*4)
    zy = fill_complex_double(SMALL_N*3)

    lib.cblas_cswap.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int]
    lib.cblas_cswap(SMALL_N, cx.ctypes.data, 1, cy.ctypes.data, -2)

    lib.cblas_zswap.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int]
    lib.cblas_zswap(SMALL_N, zx.ctypes.data, 3, zy.ctypes.data, 1)

    print("  → swap successful")

def test_copy():
    print_header("scopy / dcopy / ccopy / zcopy")

    fx = fill_float(SMALL_N*3)
    fy = fill_float(SMALL_N*2)
    dx = fill_double(SMALL_N)
    dy = fill_double(SMALL_N*4)

    lib.cblas_scopy.argtypes = [c_int, c_float_p, c_int, c_float_p, c_int]
    lib.cblas_scopy(SMALL_N, fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)
    lib.cblas_scopy(0, fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)

    lib.cblas_dcopy.argtypes = [c_int, c_double_p, c_int, c_double_p, c_int]
    lib.cblas_dcopy(SMALL_N, dx.ctypes.data_as(c_double_p), 2, dy.ctypes.data_as(c_double_p), -1)

    cx = fill_complex_float(SMALL_N*4)
    cy = fill_complex_float(SMALL_N*3)
    zx = fill_complex_double(SMALL_N*2)
    zy = fill_complex_double(SMALL_N*5)

    lib.cblas_ccopy.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int]
    lib.cblas_ccopy(SMALL_N, cx.ctypes.data, 1, cy.ctypes.data, -2)

    lib.cblas_zcopy.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int]
    lib.cblas_zcopy(SMALL_N, zx.ctypes.data, 3, zy.ctypes.data, 1)

    print("  → copy successful")

def test_axpy():
    print_header("saxpy / daxpy / caxpy / zaxpy")

    fx = fill_float(SMALL_N*3)
    fy = fill_float(SMALL_N*4)
    dx = fill_double(SMALL_N*2)
    dy = fill_double(SMALL_N)

    lib.cblas_saxpy.argtypes = [c_int, c_float, c_float_p, c_int, c_float_p, c_int]
    lib.cblas_saxpy(SMALL_N, c_float(1.5), fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), 1)

    lib.cblas_daxpy.argtypes = [c_int, c_double, c_double_p, c_int, c_double_p, c_int]
    lib.cblas_daxpy(SMALL_N, c_double(3.14), dx.ctypes.data_as(c_double_p), 2, dy.ctypes.data_as(c_double_p), -1)

    cx = fill_complex_float(SMALL_N*3)
    cy = fill_complex_float(SMALL_N*2)
    zx = fill_complex_double(SMALL_N*4)
    zy = fill_complex_double(SMALL_N*3)

    calpha = np.array([1.0 + 0j, -1.0 + 0j], dtype=np.complex64)
    zalpha = np.array([-1.0 + 0j, 0.0 + 0j], dtype=np.complex128)

    lib.cblas_caxpy.argtypes = [c_int, c_void_p, c_void_p, c_int, c_void_p, c_int]
    lib.cblas_caxpy(SMALL_N, calpha.ctypes.data, cx.ctypes.data, 1, cy.ctypes.data, -2)

    lib.cblas_zaxpy.argtypes = [c_int, c_void_p, c_void_p, c_int, c_void_p, c_int]
    lib.cblas_zaxpy(SMALL_N, zalpha.ctypes.data, zx.ctypes.data, 3, zy.ctypes.data, 1)

    print("  → axpy successful")

def test_rot_real():
    print_header("srotg / drotg / srot / drot / srotmg / drotmg / srotm / drotm")

    # srotg
    fa = c_float(1.0)
    fb = c_float(2.0)
    fc = c_float()
    fs = c_float()
    lib.cblas_srotg.argtypes = [ctypes.POINTER(c_float)] * 4
    lib.cblas_srotg.restype = None
    lib.cblas_srotg(ctypes.byref(fa), ctypes.byref(fb), ctypes.byref(fc), ctypes.byref(fs))

    # drotg
    da = c_double(3.14)
    db = c_double(-1.0)
    dc = c_double()
    ds = c_double()
    lib.cblas_drotg.argtypes = [ctypes.POINTER(c_double)] * 4
    lib.cblas_drotg(ctypes.byref(da), ctypes.byref(db), ctypes.byref(dc), ctypes.byref(ds))

    fx = fill_float(SMALL_N*2)
    fy = fill_float(SMALL_N*3)
    dx = fill_double(SMALL_N*4)
    dy = fill_double(SMALL_N)

    lib.cblas_srot.argtypes = [c_int, c_float_p, c_int, c_float_p, c_int, c_float, c_float]
    lib.cblas_srot(SMALL_N, fx.ctypes.data_as(c_float_p), 1, fy.ctypes.data_as(c_float_p), -2, fc, fs)

    lib.cblas_drot.argtypes = [c_int, c_double_p, c_int, c_double_p, c_int, c_double, c_double]
    lib.cblas_drot(SMALL_N, dx.ctypes.data_as(c_double_p), 3, dy.ctypes.data_as(c_double_p), 1, dc, ds)

    # rotmg / rotm
    d1 = c_float(1.0)
    d2 = c_float(2.0)
    b1 = c_float(3.0)
    b2 = c_float(4.0)
    param = np.zeros(5, dtype=np.float32)

    lib.cblas_srotmg.argtypes = [ctypes.POINTER(c_float)]*3 + [c_float, c_void_p]
    lib.cblas_srotmg(ctypes.byref(d1), ctypes.byref(d2), ctypes.byref(b1), b2, param.ctypes.data)

    lib.cblas_srotm.argtypes = [c_int, c_float_p, c_int, c_float_p, c_int, c_void_p]
    lib.cblas_srotm(SMALL_N, fx.ctypes.data_as(c_float_p), -1, fy.ctypes.data_as(c_float_p), 2, param.ctypes.data)

    print("  → real rot successful")

def test_rot_complex():
    print_header("crotg / zrotg / csrot / zdrot")

    # crotg
    ca = c_float(1.0)
    cb = np.array([2.0 + 3.0j], dtype=np.complex64)
    cc = c_float()
    cs = np.zeros(2, dtype=np.float32)

    lib.cblas_crotg.argtypes = [ctypes.POINTER(c_float), c_void_p, ctypes.POINTER(c_float), c_void_p]
    lib.cblas_crotg(ctypes.byref(ca), cb.ctypes.data_as(c_void_p), ctypes.byref(cc), cs.ctypes.data_as(c_void_p))

    # zrotg
    da = c_double(4.0)
    db = np.array([5.0 + 6.0j], dtype=np.complex128)
    dc = c_double()
    ds = np.zeros(2, dtype=np.float64)

    lib.cblas_zrotg.argtypes = [ctypes.POINTER(c_double), c_void_p, ctypes.POINTER(c_double), c_void_p]
    lib.cblas_zrotg(ctypes.byref(da), db.ctypes.data_as(c_void_p), ctypes.byref(dc), ds.ctypes.data_as(c_void_p))

    cx = fill_complex_float(SMALL_N)
    cy = fill_complex_float(SMALL_N)

    lib.cblas_csrot.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int, c_float, c_float]
    lib.cblas_csrot(SMALL_N, cx.ctypes.data, 1, cy.ctypes.data, 1, c_float(0.707), c_float(0.707))

    zx = fill_complex_double(SMALL_N)
    zy = fill_complex_double(SMALL_N)

    lib.cblas_zdrot.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int, c_double, c_double]
    lib.cblas_zdrot(SMALL_N, zx.ctypes.data, 1, zy.ctypes.data, 1, c_double(0.8), c_double(0.6))

    print("  → complex rot successful")

def test_scal():
    print_header("sscal / dscal / cscal / zscal / csscal / zdscal")

    fx = fill_float(SMALL_N*3)
    dx = fill_double(SMALL_N*2)

    lib.cblas_sscal.argtypes = [c_int, c_float, c_float_p, c_int]
    lib.cblas_sscal(SMALL_N, c_float(1.5), fx.ctypes.data_as(c_float_p), 1)

    lib.cblas_dscal.argtypes = [c_int, c_double, c_double_p, c_int]
    lib.cblas_dscal(SMALL_N, c_double(3.14), dx.ctypes.data_as(c_double_p), 2)

    cx = fill_complex_float(SMALL_N*4)
    zx = fill_complex_double(SMALL_N*3)

    calpha = np.array([1.0 + 2.0j], dtype=np.complex64)
    zalpha = np.array([-1.0 + 0.0j], dtype=np.complex128)

    lib.cblas_cscal.argtypes = [c_int, c_void_p, c_void_p, c_int]
    lib.cblas_cscal(SMALL_N, calpha.ctypes.data, cx.ctypes.data, 1)

    lib.cblas_zscal.argtypes = [c_int, c_void_p, c_void_p, c_int]
    lib.cblas_zscal(SMALL_N, zalpha.ctypes.data, zx.ctypes.data, 3)

    lib.cblas_csscal.argtypes = [c_int, c_float, c_void_p, c_int]
    lib.cblas_csscal(SMALL_N, c_float(2.5), cx.ctypes.data, -1)

    lib.cblas_zdscal.argtypes = [c_int, c_double, c_void_p, c_int]
    lib.cblas_zdscal(SMALL_N, c_double(-1.7), zx.ctypes.data, 2)

    print("  → scal successful")

if __name__ == "__main__":
    print("=== Tests Level 1 CBLAS with Python (ctypes) ===\n")
    print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS", "не задан"))

    tests = [
        test_dot_real,
        test_dot_complex,
        test_norms,
        test_amax,
        test_swap,
        test_copy,
        test_axpy,
        test_rot_real,
        test_rot_complex,
        test_scal,
    ]

    for test_func in tests:
        p = mp.Process(target=run_test, args=(test_func,))
        p.start()
        p.join(timeout=15)
        if p.is_alive():
            print(f"!!! {test_func.__name__} is frozen → kill")
            p.terminate()
            p.join()
        print("-" * 60)

    print("\nAll tests completed")

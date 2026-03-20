# SDT  
(учебная реализация и тестирование BLAS-подобных операций)

Лёгкий проект для:

- понимания и тестирования интерфейса **CBLAS** (уровень 1)
- экспериментов с реализацией **symm** (SSYMM/DSYMM) — симметричное матрично-матричное умножение
- сравнения производительности самописного кода с OpenBLAS

## Что уже есть

- **cblas_level1_tests.c** — python-обвязка через `ctypes` для проверки вызовов CBLAS level 1  
  (sdot, ddot, snrm2, scnrm2, isamax, sswap, saxpy, sscal, crotg, csrot и др.)
- **my-blas-symm** — начальная реализация `ssymm` / `dsymm` с блочным подходом (blocking)
- **benchmark.cpp** — простой бенчмарк сравнения `my_ssymm` vs `cblas_ssymm` (OpenBLAS) на разных числах потоков
- Makefile (минимальный)

## Текущий статус

🚧 **очень ранняя работа в процессе**  
Это больше учебный проект, чем библиотека.

## Как собрать и запустить

### Требования

- gcc / clang (C11 / C++11)
- OpenBLAS
- python3 + numpy (для запуска тестов level 1)

### Сборка

```bash
# Базовая сборка (gcc)
make

# С OpenBLAS (самый полезный вариант)
make USE_OPENBLAS=1

# С Intel MKL (если установлен oneAPI)
make USE_MKL=1

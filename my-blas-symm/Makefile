CXX = g++
CXXFLAGS = -O3 -march=native -mtune=native -std=c++17 -Wall

all: benchmark

benchmark: tests/benchmark.cpp src/symm.cpp
	$(CXX) $(CXXFLAGS) -I/usr/local/include -I/usr/local/include/openblas \
            -L/usr/local/lib $^ -o $@ -lopenblas -lpthread

clean:
	rm -f benchmark
# Sulimovici Raoul-Renatto 331CB

## Exercitiul 1
GFLOPS: 11.800

## Exercitiul 2
Timing simple implementation... done.

Timing optimized implementation... done.

Matrix size: 1024x1024

Tile size: 16x16

Throughput of simple kernel: 80366.1 GFLOPS

Throughput of optimized kernel: 162385 GFLOPS

Performance improvement: 2.02056x

## Profiling
## nvprof
### matrix_multiplication
66% din timp este petrecut in matrix_multiply_simple

33% in matrix_multiply
### task_gflops
5% este petrecut in kernel_gflops

94% memcpy host to device
## nvpp

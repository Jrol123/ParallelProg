#include<iostream>
#include<cuda_runtime.h>
using namespace std;

const short N = 5;

// Kernel definition
global void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
int main()
{
    float matrixA[N] = [1, 1, 1, 1, 1];
    float matrixB[N] = [1, 2, 3, 4, 5];
    float matrixC[N];

    float *dev_A, *dev_B, *dev_C;
    cudaMalloc( (void**)&dev_A, sizeof(float)*N );
    cudaMalloc( (void**)&dev_B, sizeof(float)*N );
    cudaMalloc( (void**)&dev_C, sizeof(float)*N );
    
    cudaMemcpy( dev_A, matrixA, sizeof(float)*N, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_B, matrixB, sizeof(float)*N, cudaMemcpyHostToDevice );

    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(matrixA, matrixB, matrixC);

    for (int i = 0; i < N; i++)
    {
        cout << matrixC[i] << endl;
    }

    cudaDeviceSynchronize();

    return 0;
}
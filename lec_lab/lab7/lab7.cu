#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fstream> // Для работы с файлами

#define EPS (0.001)
#define N (4 * 1)
#define STEPS (100)
#define KERNEL (4)

using namespace std;

__global__ void integrateBodies(float3 *newPos, float3 *newVel, float3 *oldPos, float3 *oldVel, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float3 pos = oldPos[index];
    float3 f = make_float3(0.0, 0.0, 0.0);

    for (int i = 0; i < N; ++i)
    {
        float3 pi = oldPos[i];
        float3 r;

        r.x = pi.x - pos.x;
        r.y = pi.y - pos.y;
        r.z = pi.z - pos.z;

        float invDist = 1.0 / sqrtf(r.x * r.x + r.y * r.y + r.z * r.z + EPS * EPS);
        float s = invDist * invDist * invDist;

        f.x += r.x * s;
        f.y += r.y * s;
        f.z += r.z * s;
    }

    float3 vel = oldVel[index];

    vel.x += f.x * dt;
    vel.y += f.y * dt;
    vel.z += f.z * dt;

    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    newPos[index] = pos;
    newVel[index] = vel;
}

void randomInit(float3 *a, int n)
{
    for (int i = 0; i < n; ++i)
    {
        a[i].x = rand() / (float)RAND_MAX - 0.5;
        a[i].y = rand() / (float)RAND_MAX - 0.5;
        a[i].z = rand() / (float)RAND_MAX - 0.5;
        std::cout << a[i].x << " " << a[i].y << " " << a[i].z << "\n";
    }
    std::cout << "\n";
}

void writeToFile(float3 *p, float3 *v, int step)
{
    ofstream outFile;
    outFile.open("output.csv", ios::app); // Открываем файл для добавления данных
    if (step == 1)
    {
        // Записываем заголовки в первый шаг
        outFile << "Step,Point,Position X,Position Y,Position Z,Velocity X,Velocity Y,Velocity Z\n";
    }
    for (int i = 0; i < N; ++i)
    {
        outFile << step << ","
                << i << ","
                << p[i].x << ","
                << p[i].y << ","
                << p[i].z << ","
                << v[i].x << ","
                << v[i].y << ","
                << v[i].z << "\n";
    }
    outFile.close();
}

int main()
{
    float3 *p = new float3[N];
    float3 *v = new float3[N];

    float3 *pDev[2] = {NULL, NULL};
    float3 *vDev[2] = {NULL, NULL};

    cudaEvent_t start, stop;

    int index = 0;

    float gpuTime = 0.0;

    randomInit(p, N);
    randomInit(v, N);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void **)&pDev[0], N * sizeof(float3));
    cudaMalloc((void **)&pDev[1], N * sizeof(float3));
    cudaMalloc((void **)&vDev[0], N * sizeof(float3));
    cudaMalloc((void **)&vDev[1], N * sizeof(float3));

    cudaMemcpy(pDev[0], p, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(vDev[0], v, N * sizeof(float3), cudaMemcpyHostToDevice);


    for (int i = 0; i < STEPS; i++, index ^= 1)
    {
        integrateBodies<<<dim3(N / KERNEL), dim3(KERNEL)>>>(pDev[index ^ 1], vDev[index ^ 1], pDev[index], vDev[index], 0.01);

        // Копирование новых позиций и скоростей обратно на хост
        cudaMemcpy(p, pDev[index ^ 1], N * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, vDev[index ^ 1], N * sizeof(float3), cudaMemcpyDeviceToHost);

        // Запись текущих позиций и скоростей в файл
        writeToFile(p, v, i + 1);
    }

    cudaMemcpy(p, pDev[index ^ 1], N * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, vDev[index ^ 1], N * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(pDev[0]);
    cudaFree(pDev[1]);
    cudaFree(vDev[0]);
    cudaFree(vDev[1]);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("Elapsed time: %.2f\n", gpuTime);

    delete p,
        delete v;

    return 0;
}

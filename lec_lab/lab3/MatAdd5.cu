#include <iostream>
#include <cuda_runtime.h>

using namespace std;

const int N = 2; // Количество строк
const int M = 3; // Количество столбцов

// Kernel definition
__global__ void MatAdd(float A[N][M], float B[N][M], float C[N][M])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Индекс потока
    if (idx < N * M)
    {
        int i = idx / M;             // Получаем номер строки
        int j = idx % M;             // Получаем номер столбца
        C[i][j] = A[i][j] * B[i][j]; // Скалярное произведение
    }
}

__global__ void Reduction(float C[N][M], *float result)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Индекс потока
    // Считать матрицу как линию
    if(idx < N * M / counter)
}

int main()
{
    // Выделение двумерного массива
    float(*A)[M] = new float[N][M]; // матрица A
    float(*B)[M] = new float[N][M]; // матрица B
    float(*C)[M] = new float[N][M]; // матрица C
    float h_result = 0.0f;          // Результат на хосте

    // Инициализация массивов
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            A[i][j] = static_cast<float>(i * j);
            B[i][j] = static_cast<float>(i * j);
        }
    }

    float(*dev_A)[M], (*dev_B)[M], (*dev_C)[M]; // Указатели на массивы на устройстве
    float *dev_result;                          // Указатель на результат на устройстве

    // Выделение памяти на устройстве
    cudaMalloc((void **)&dev_A, N * M * sizeof(float));
    cudaMalloc((void **)&dev_B, N * M * sizeof(float));
    cudaMalloc((void **)&dev_C, N * M * sizeof(float));
    cudaMalloc((void **)&dev_result, sizeof(float)); // Для итоговой суммы

    // Копирование данных на устройство
    cudaMemcpy(dev_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, &h_result, sizeof(float), cudaMemcpyHostToDevice); // Сброс итогового результата

    // Запуск ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * M + threadsPerBlock - 1) / threadsPerBlock;
    MatAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C);
    Reduction<<<blocksPerGrid, threadsPerBlock>>>(devC, dev_result);

    // Копирование результата обратно на хост
    cudaMemcpy(&h_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&C, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            cout << C[i][j] << ' ';
        }
        cout << '\n';
    }

    // Вывод результатов
    cout << "Скалярное произведение: " << h_result << endl;

    // Освобождение ресурсов
    delete[] A;
    delete[] B;
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFree(dev_result);

    return 0;
}

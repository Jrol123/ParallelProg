#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <vector>
using namespace std;

const int MAIN_PROCESS = 0;
const int TOTAL_SIZE = 3;

void output_matrix(int matrix[TOTAL_SIZE][TOTAL_SIZE], int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char **argv)
{
    // TODO: Распихивать матрицу на проциональное количество потоков
    MPI_Init(&argc, &argv);
    int rank, size;
    int sid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_count = 0;

    int matrixA[TOTAL_SIZE][TOTAL_SIZE];
    int matrixB[TOTAL_SIZE][TOTAL_SIZE];
    int end_matrix[TOTAL_SIZE][TOTAL_SIZE];

    srand(1 + sid);

    if (rank == MAIN_PROCESS)
    {
        for (int i = 0; i < TOTAL_SIZE; i++)
        {
            for (int j = 0; j < TOTAL_SIZE; j++)
            {
                matrixA[i][j] = 1 + ((double)rand() / RAND_MAX) * 100;
                matrixB[i][j] = 1 + ((double)rand() / RAND_MAX) * 100;
            }
        }
        output_matrix(matrixA, TOTAL_SIZE);
        output_matrix(matrixB, TOTAL_SIZE);

        // Последовательное

        double start_time = MPI_Wtime();

        for (int i = 0; i < TOTAL_SIZE; i++)
        {
            for (int j = 0; j < TOTAL_SIZE; j++)
            {
                double time_sum = 0;
                for (int c = 0; c < TOTAL_SIZE; c++)
                {
                    time_sum += matrixA[i][c] * matrixB[c][j];
                }
                end_matrix[i][j] = time_sum;
            }
        }

        double end_time = MPI_Wtime();
        cout << "Difference in time: " << end_time - start_time << endl;
        output_matrix(end_matrix, TOTAL_SIZE);
    }

    double start_time = MPI_Wtime();

    MPI_Scatterv();

    double end_time = MPI_Wtime();
    
    cout << "Difference in time: " << end_time - start_time << endl;
    output_matrix(end_matrix, TOTAL_SIZE);


    MPI_Finalize();
}
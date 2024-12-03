#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <vector>
using namespace std;

const bool PRINT = true;

const int MAIN_PROCESS = 0;
const int TOTAL_SIZE = 2;

void output_matrix(vector<int> &matrix)
{
    for (int i = 0; i < TOTAL_SIZE; i++)
    {
        for (int j = 0; j < TOTAL_SIZE; j++)
        {
            cout << matrix[i * TOTAL_SIZE + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Function to multiply matrices in parallel
void multiply_matrices_parallel(int rank, int size, const std::vector<int> &matrixA, const std::vector<int> &matrixB, std::vector<int> &local_result, int rows_for_process)
{
    for (int i = 0; i < rows_for_process; i++)
    {
        for (int j = 0; j < TOTAL_SIZE; j++)
        {
            local_result[i * TOTAL_SIZE + j] = 0;
            for (int k = 0; k < TOTAL_SIZE; k++)
            {
                local_result[i * TOTAL_SIZE + j] += matrixA[i * TOTAL_SIZE + k] * matrixB[k * TOTAL_SIZE + j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    int sid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_count = 0;

    vector<int> matrixA(TOTAL_SIZE * TOTAL_SIZE);
    vector<int> matrixB(TOTAL_SIZE * TOTAL_SIZE);
    vector<int> end_matrixS(TOTAL_SIZE * TOTAL_SIZE);
    vector<int> end_matrixP(TOTAL_SIZE * TOTAL_SIZE);

    srand(1 + sid);

    if (rank == MAIN_PROCESS)
    {
        for (int i = 0; i < TOTAL_SIZE; i++)
        {
            for (int j = 0; j < TOTAL_SIZE; j++)
            {
                matrixA[i * TOTAL_SIZE + j] = 1 + ((double)rand() / RAND_MAX) * 10;
                matrixB[i * TOTAL_SIZE + j] = 1 + ((double)rand() / RAND_MAX) * 10;
            }
        }
        if (PRINT)
        {
            output_matrix(matrixA);
            output_matrix(matrixB);
        }

        // Последовательное

        double start_time = MPI_Wtime();

        for (int i = 0; i < TOTAL_SIZE; i++)
        {
            for (int j = 0; j < TOTAL_SIZE; j++)
            {
                double time_sum = 0;
                for (int c = 0; c < TOTAL_SIZE; c++)
                {
                    time_sum += matrixA[i * TOTAL_SIZE + c] * matrixB[c * TOTAL_SIZE + j];
                }
                end_matrixS[i * TOTAL_SIZE + j] = time_sum;
            }
        }

        double end_time = MPI_Wtime();
        cout << "Difference in time: " << end_time - start_time << endl;
        if (PRINT)
            output_matrix(end_matrixS);
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(matrixB.data(), TOTAL_SIZE * TOTAL_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = TOTAL_SIZE / size;
    int extra_rows = TOTAL_SIZE % size;
    int rows_for_process = rows_per_process + (rank < extra_rows ? 1 : 0);

    // Allocate local matrices
    vector<int> A_sub(rows_for_process * TOTAL_SIZE);
    vector<int> C_local(rows_for_process * TOTAL_SIZE); // Local result matrix

    // Scatter matrix A
    int sendcounts[size];
    int displacements[size];
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = rows_per_process * TOTAL_SIZE;
        if (i < extra_rows)
        {
            sendcounts[i] += TOTAL_SIZE; // Give extra rows to the first 'extra_rows' processes
        }
        displacements[i] = (i * rows_per_process + (i < extra_rows ? i : extra_rows)) * TOTAL_SIZE;
    }

    double start_time = MPI_Wtime();
    MPI_Scatterv(matrixA.data(), sendcounts, displacements, MPI_INT, A_sub.data(), rows_for_process * TOTAL_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local multiplication
    multiply_matrices_parallel(rank, size, A_sub, matrixB, C_local, rows_for_process);

    // Gather results
    MPI_Gatherv(C_local.data(), rows_for_process * TOTAL_SIZE, MPI_INT, end_matrixP.data(), sendcounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == MAIN_PROCESS)
    {
        cout << "Difference in time: " << end_time - start_time << endl;
        if (PRINT)
            output_matrix(end_matrixP);
    }

    MPI_Finalize();
}
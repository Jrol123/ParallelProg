#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <vector>
using namespace std;

const int MAIN_PROCESS = 0;
const int TOTAL_SIZE = 100000000;

const bool PRINT = false;

void generate_vector(vector<int> &vector)
{
    for (int i = 0; i < TOTAL_SIZE; i++)
    {
        vector[i] = 1 + ((double)rand() / RAND_MAX) * 100;
    }
}

void output_vector(vector<int> &vector)
{
    for (int i = 0; i < TOTAL_SIZE; i++)
    {
        cout << vector[i] << " ";
    }
    cout << endl;
}

int find_max(const std::vector<int> &arr)
{
    int max_element = arr[0];
    for (int i = 1; i < arr.size(); i++)
    {
        if (arr[i] > max_element)
        {
            max_element = arr[i];
        }
    }
    return max_element;
}

int main(int argc, char **argv)
{
    int sid = 0;
    srand(1 + sid);

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> main_vector(TOTAL_SIZE);
    if (rank == MAIN_PROCESS)
    {
        generate_vector(main_vector);
        if (PRINT)
        {
            output_vector(main_vector);
            cout << endl;
        }

        cout << "Sequential" << endl
             << endl;
        double start_time = MPI_Wtime();

        int max_element = find_max(main_vector);

        double end_time = MPI_Wtime();

        if (PRINT)
            cout << "Max element: " << max_element << endl;

        cout << "Difference in time: " << end_time - start_time << endl << endl;
    }

    int base_size = TOTAL_SIZE / size;
    int remained_count = TOTAL_SIZE % size;

    int elements_processed = base_size + (rank < remained_count ? 1 : 0);
    vector<int> arr_sub(elements_processed);

    vector<int> sendcounts(size, base_size);
    vector<int> displs(size, 0);

    for (int i = 0; i < remained_count; i++)
    {
        sendcounts[i] += 1;
    }

    for (int i = 1; i < size; i++)
    {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    // cout << "Sequential" << endl << endl;
    double start_time = MPI_Wtime();

    MPI_Scatterv(main_vector.data(), sendcounts.data(), displs.data(), MPI_INT,
                 arr_sub.data(), elements_processed, MPI_INT, 0, MPI_COMM_WORLD);

    int local_max = find_max(arr_sub);

    int global_max;

    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        cout << "Parallel" << endl << endl;
        if(PRINT)
            cout << "Max element: " << global_max << endl;

        cout << "Difference in time: " << end_time - start_time << endl;
    }

    MPI_Finalize();
}
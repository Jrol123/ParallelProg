#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
using namespace std;

const int MAIN_PROCESS = 0;
const int COUNT_ELEMENT = 10;

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int loc_max, global_max;
    int loc_sum, global_sum;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int mass_data[COUNT_ELEMENT];

    // Инициализация массива для основного процесса
    if (rank == MAIN_PROCESS)
    {
        for (int i = 0; i < COUNT_ELEMENT; i++)
        {
            mass_data[i] = i + 1;
        }
        cout << "Process " << MAIN_PROCESS << " started working..." << endl;
    }
    // Отправка и принятие различными процессами инфы
    MPI_Bcast(mass_data, COUNT_ELEMENT, MPI_INT, MAIN_PROCESS, MPI_COMM_WORLD);

    // Обработка данных в побочном процессе
    if (rank != MAIN_PROCESS)
    {
        for (int i = 0; i < COUNT_ELEMENT - 1; i++)
        {
            loc_sum += mass_data[i] * rank;
        }
        loc_max = mass_data[COUNT_ELEMENT - 1] * rank;
        loc_sum += loc_max;

        // cout << "Process number: " << rank << "\tHas max number: " << loc_max << endl;
    }

    MPI_Reduce(&loc_sum, &global_sum, 1, MPI_INT, MPI_SUM, MAIN_PROCESS, MPI_COMM_WORLD);
    MPI_Reduce(&loc_max, &global_max, 1, MPI_INT, MPI_MAX, MAIN_PROCESS, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == MAIN_PROCESS)
    {
        cout << "Total sum of all elements: " << global_sum << endl;
        cout << "Total max of all elements: " << global_max << endl;
    }

    MPI_Finalize();
    return 0;
}
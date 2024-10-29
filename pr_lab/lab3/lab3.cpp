#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
using namespace std;

const int MAIN_PROCESS = 0;
const int SEND_COUNT = 4;

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int mass_data[SEND_COUNT * size];
    int local_data[SEND_COUNT];
    int local_sum = 0, global_sum;

    int recv_data[SEND_COUNT * size];

    if(rank == MAIN_PROCESS)
    {
        cout << "Process " << rank << " data: ";
        for(int i = 0; i < SEND_COUNT * size; i++)
        {
            mass_data[i] = i + 1;
            cout << i + 1 << " ";
        }

        cout << endl;
        
    }

    MPI_Scatter(mass_data, SEND_COUNT, MPI_INT, local_data, SEND_COUNT, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i = 0; i < SEND_COUNT; i++){
        local_data[i] *= rank;
        local_sum += local_data[i];
    }

    MPI_Gather(local_data, SEND_COUNT, MPI_INT, recv_data, SEND_COUNT, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == MAIN_PROCESS)
    {
        cout << "Process " << rank << " gathered data: ";
        for(int i = 0; i < SEND_COUNT * size; i++)
        {
            cout << recv_data[i] << " ";
        }

        cout << endl;
    }

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    cout << "Process " << rank << " total sum: " << global_sum << endl;

    MPI_Finalize();

    return 0;
}
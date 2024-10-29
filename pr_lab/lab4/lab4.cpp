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

    // Синхрон

    double start_time = MPI_Wtime();

    //! Засечь время
    MPI_Scatter(mass_data, SEND_COUNT, MPI_INT, local_data, SEND_COUNT, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i = 0; i < SEND_COUNT; i++){
        local_data[i] *= rank;
    }

    MPI_Gather(local_data, SEND_COUNT, MPI_INT, recv_data, SEND_COUNT, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == MAIN_PROCESS)
    {
        double end_time = MPI_Wtime();

        cout << "Process " << rank << " gathered data: ";
        for(int i = 0; i < SEND_COUNT * size; i++)
        {
            cout << recv_data[i] << " ";
        }
        cout << endl;
        cout << "Difference in time: " << end_time - start_time << endl;
    }
    // double time12 = MPI_Wtime();
    // cout << time12 << endl;

    int local_data2[SEND_COUNT];
    int recv_data2[SEND_COUNT * size];

    // Асинхрон
    MPI_Request request;

    start_time = MPI_Wtime();

    if(rank == MAIN_PROCESS)
    {
        cout << "Process " << rank << " data: ";
        for(int i = 0; i < SEND_COUNT * size; i++)
        {
            mass_data[i] = i + 1;
            cout << mass_data[i] << " ";
        }

        cout << endl;
        
    }

    MPI_Iscatter(mass_data, SEND_COUNT, MPI_INT, local_data2, SEND_COUNT, MPI_INT, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    for(int i = 0; i < SEND_COUNT; i++){
        // cout << local_data[i];
        local_data2[i] *= rank;
        // cout << " " << local_data[i] << " ";
    }


    MPI_Igather(local_data2, SEND_COUNT, MPI_INT, recv_data2, SEND_COUNT, MPI_INT, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    if(rank == MAIN_PROCESS)
    {
        double end_time = MPI_Wtime();
        cout << "Process " << rank << " gathered data: ";
        for(int i = 0; i < SEND_COUNT * size; i++)
        {
            cout << recv_data2[i] << " ";
        }

        cout << endl;

        cout << "Difference in time: " << end_time - start_time << endl;
    }

    MPI_Finalize();

    return 0;
}
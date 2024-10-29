#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>

const int MAIN_PROCESS = 0;

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string message;

    if(rank == MAIN_PROCESS){
        for (int i = 1; i < size; ++i){
            message = "0x0";
            std::cout << "Process number " << rank << " sent message: " << message << " to process " << i << std::endl; 
            MPI_Send(message.c_str(), message.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }
    else{
        char buffer[256];
        MPI_Recv(buffer, sizeof(buffer), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process number " << rank << " recieved message: " << buffer << " from process " << MAIN_PROCESS << std::endl;
    }

    MPI_Finalize();
    return 0;
}
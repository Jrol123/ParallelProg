#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
using namespace std;

const int MAIN_PROCESS = 0;
const int GENERATE_COUNT = 100000;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_count = 0;

    srand(1 + rank);
    int count = 0;

    double start_time = MPI_Wtime();

    for(int i = 0; i < GENERATE_COUNT; i++){
        double x = (double)rand() / RAND_MAX; 
        double y = (double)rand() / RAND_MAX; 

        // cout << x << " " << y << endl;

        if ((x * x + y * y) <= 1){
            count += 1;
        }
    }

    MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("Result: %lf", ((double)total_count / (GENERATE_COUNT * size)) * 4);
        // cout << "Result: " <<  << endl;
        double end_time = MPI_Wtime();
        cout << "Difference in time: " << end_time - start_time << endl;
    }   

    MPI_Finalize();
}
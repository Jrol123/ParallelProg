#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
using namespace std;

const int MAIN_PROCESS = 0;
const int TOTAL_SIZE = 5;

void output_matrix(int matrix[TOTAL_SIZE][TOTAL_SIZE], int size){
    for(int i = 0; i < size; i ++){
        for(int j = 0; j < size; j++){
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    int sid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_count = 0;

    int main_matrix [TOTAL_SIZE][TOTAL_SIZE];

    srand(1 + sid);

    if(rank == MAIN_PROCESS){

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            main_matrix[i][j] = 1 + ((double)rand() / RAND_MAX) * 100;
        }
    }
        output_matrix(main_matrix, size);       
    }

    MPI_Finalize();
}
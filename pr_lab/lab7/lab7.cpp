#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <vector>
using namespace std;

const int MAIN_PROCESS = 0;
const int TOTAL_SIZE = 10;

void generate_vector(vector<int> &vector)
{
    for (int i = 0; i < TOTAL_SIZE; i++)
    {
        vector[i] = (rand());
    }
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
        for (int i = 0; i < TOTAL_SIZE; i++)
        {
            cout << main_vector[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
}
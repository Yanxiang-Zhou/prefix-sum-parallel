#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unistd.h>

/*
    write_array_by_rank() prints data in the format below, ordered by rank.

    rank {rank} {description}: {array[0]}, {array[1]}, ..., {array[array_len-1]}

    Note: this is a collective operation, meaning all processes must call this function.
*/
void write_array_by_rank(MPI_Comm comm, const char* description, const int* array, int array_len);


#endif // UTILS_H

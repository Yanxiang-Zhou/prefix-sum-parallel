#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unistd.h>

void write_array_by_rank(MPI_Comm comm, const char* description, const int* array, int array_len);


#endif // UTILS_H

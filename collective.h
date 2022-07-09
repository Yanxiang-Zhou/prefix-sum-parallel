#ifndef COLLECTIVE_H
#define COLLECTIVE_H

#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>
#include <mpi.h>
#include <unistd.h>


typedef const void* (HPC_Prefix_func) (const void* pre, const void* a, void* b, int len);



void HPC_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);



void HPC_Prefix(const HPC_Prefix_func* func, const void *sendbuf, void *recvbuf, int count,
                MPI_Datatype datatype, MPI_Comm comm, void* wb1, void* wb2, void* wb3);


#endif // COLLECTIVE_H

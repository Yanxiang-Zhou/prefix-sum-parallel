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

/*
    HPC_Prefix_func represents a function that computes the prefix array for a binary associative operator.
    It should return the final value of the prefix array.

    The implementation should look like this, for a binary associative operator f:

        if (pre != NULL) b[0] = f(pre, a[0]);
        else b[0] = a[0];

        for (int i = 1; i < len; ++i) {
            b[i] = f(a[i], b[i-1]);
        }
        return &b[len-1];

    Arguments:
        pre: initial prefix to apply. Ignored if NULL.
        a: input array
        b: output prefix array (may be equal to a)
        len: length of a and b.


    // Here is an example function definition.

        int sum(int a, int b) { return a + b; }
        void myDoublePrefixSum(const void* pre_v, const void* a_v, void* b_v, int len) {
            double* pre = (double*) pre_v;
            double* a = (double*) a_v;
            double* b = (double*) b_v;

            if (pre != NULL) b[0] = sum(*pre, a[0]);
            else b[0] = a[0];

            for (int i = 1; i < len; ++i) {
                b[i] = sum(b[i-1], a[i]);
            }
            return &b[len-1];
        }

    // And here is how that function could be called.

        HPC_Prefix_func func = myAdd;
        double a, b;

        a = 3.0;
        func(NULL, &a, &b, 1);
        assert(b == 3.0);

        b = 10.0;
        func(&a, &b, &b, 1);
        assert(b == 13.0);

        double aa[2], bb[2];
        aa[0] = 1.0;
        aa[1] = 2.0;
        func(NULL, aa, bb, 2);
        assert(bb[0] == 1.0);
        assert(bb[0] == 3.0);

        c = 10.0;
        func(&c, aa, bb, 2);
        assert(bb[0] == 11.0);
        assert(bb[0] == 13.0);
*/
typedef const void* (HPC_Prefix_func) (const void* pre, const void* a, void* b, int len);


/*
    HPC_Bcast should implement broadcast on a hypercubic connection network.
    The parameters are equivalent to those for MPI_Bcast.
*/
void HPC_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);


/*
    HPC_Prefix should implement parallel prefix on a hypercubic connection network.

    Arguments:
        sendbuf: an array of `count` elements of type `MPI_Datatype`.

        recvbuf: an array of `count` elements of type `MPI_Datatype`. The result
                 of the prefix operation will be stored here.

        count: the number of elements in the input array.

        datatype: the datatype of the input and output data elements.

        func: a function that runs the prefix operator on a given array.

        comm: the communicator that runs the parallel prefix.

        wb1, wb2, wb3: work buffers that can be used as temporary storage in the
                       HPC_Prefix implementation. Each can hold exactly 1 value
                       of type `datatype`. A value can be copied into a buffer
                       by using the `func` operator. For example, the line of
                       code below would copy the first element of sendbuf into
                       wb1.

                           func(NULL, sendbuf, wb1, 1);
*/
void HPC_Prefix(const HPC_Prefix_func* func, const void *sendbuf, void *recvbuf, int count,
                MPI_Datatype datatype, MPI_Comm comm, void* wb1, void* wb2, void* wb3);


#endif // COLLECTIVE_H

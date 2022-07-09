#include <mpi.h>
#include "gtest-mpi-listener/gtest-mpi-listener.hpp"

#include <math.h>
#include "collective.h"
#include "utils.h"

const int TEST_DEBUG = 0;

class BcastTest : public testing::TestWithParam< std::tuple<int, int, int, int> > { };

TEST_P(BcastTest, BcastN)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    std::vector<int> buffer;
    int i, n;
    int world_size, comm_size;
    int root, rank;
    int test_number;
    char cbuf[100];

    std::tie(test_number, comm_size, root, n) = GetParam();

    sprintf(cbuf, "1.%d", test_number);
    RecordProperty("Points", "0.5");
    RecordProperty("Number", cbuf);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < comm_size) {
        GTEST_SKIP() << "Skipping test with comm_size=" << comm_size << " because MPI was run with " << world_size << " processes";
    }
    MPI_Comm_split(MPI_COMM_WORLD, rank < comm_size, rank, &comm);
    if (rank < comm_size) {
        MPI_Comm_rank(comm, &rank);

        buffer.resize(n);
        if (rank == root) {
            for (i = 0; i < n; ++i) buffer[i] = 2*i*i + 1;
        }
        else {
            for (i = 0; i < n; ++i) buffer[i] = -rank;
        }
        HPC_Bcast(&buffer[0], n, MPI_INT, root, comm);
        for (i = 0; i < n; ++i) {
            ASSERT_EQ(buffer[i], 2*i*i + 1) << " buffer[" << i << "] is wrong on rank " << rank;
        }
    }
    MPI_Comm_free(&comm);
}


INSTANTIATE_TEST_SUITE_P(BcastInt_ZeroRoot,
                         BcastTest,
                         testing::Values(
                            /* The first parameter is for Gradescope,
                               the second parameter is the communicator size,
                               the third parameter is the root,
                               and the fourth parameter is the array size.*/
                            std::tuple<int, int, int, int>{1, 4, 0, 1},
                            std::tuple<int, int, int, int>{2, 4, 0, 2},
                            std::tuple<int, int, int, int>{3, 16, 0, 43},
                            std::tuple<int, int, int, int>{4, 8, 0, 29989}
                        ));


INSTANTIATE_TEST_SUITE_P(BcastInt_NonzeroRoot,
                         BcastTest,
                         testing::Values(
                            /* The first parameter is for Gradescope,
                               the second parameter is the communicator size,
                               the third parameter is the root,
                               and the fourth parameter is the array size.*/
                            std::tuple<int, int, int, int>{5, 4, 1, 1},
                            std::tuple<int, int, int, int>{6, 16, 10, 2},
                            std::tuple<int, int, int, int>{7, 8, 2, 43},
                            std::tuple<int, int, int, int>{8, 8, 5, 29989}
                        ));



class PrefixSumTest : public testing::TestWithParam< std::tuple<int, int, int> > { };


/* This operator returns the sum of the two input numbers. */
int myIntAdd(int l, int r) { return l + r; }
const void* myIntPrefixAdd(const void* pre_v, const void* a_v, void* b_v, int len) {
    int* pre = (int*) pre_v;
    int* a = (int*) a_v;
    int* b = (int*) b_v;

    if (pre != NULL) b[0] = myIntAdd(*pre, a[0]);
    else b[0] = a[0];

    for (int i = 1; i < len; ++i) {
        b[i] = myIntAdd(b[i-1], a[i]);
    }
    return &b[len-1];
}


TEST_P(PrefixSumTest, Add)
{
    MPI_Comm comm;
    int i, n;
    int rank;
    int world_size, comm_size;
    HPC_Prefix_func *prefix_func;
    int test_number;
    char cbuf[100];

    std::tie(test_number, comm_size, n) = GetParam();

    sprintf(cbuf, "2.%d", test_number);
    RecordProperty("Points", "0.5");
    RecordProperty("Number", cbuf);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < comm_size) {
        GTEST_SKIP() << "Skipping test with comm_size=" << comm_size << " because MPI was run with " << world_size << " processes";
    }
    MPI_Comm_split(MPI_COMM_WORLD, rank < comm_size, rank, &comm);
    if (rank < comm_size) {
        MPI_Comm_rank(comm, &rank);
        prefix_func = &myIntPrefixAdd;

        std::vector<int> sendbuffer(n), recvbuffer(n);
        for (i = rank*n; i < (rank+1)*n; ++i) sendbuffer[i-rank*n] = i;

        if (TEST_DEBUG > 0 and n < 20) {
            write_array_by_rank(comm, "sendbuffer", sendbuffer.data(), n);
            MPI_Barrier(comm);
            if (rank == 0) {
                printf("\n");
                fflush(stdout);
            }
        }

        int wb1, wb2, wb3;
        HPC_Prefix(prefix_func, &sendbuffer[0], &recvbuffer[0], n, MPI_INT, comm, &wb1, &wb2, &wb3);

        if (TEST_DEBUG > 0 and n < 20) {
            write_array_by_rank(comm, "recvbuffer", recvbuffer.data(), n);
            MPI_Barrier(comm);
            if (rank == 0) {
                printf("\n");
                fflush(stdout);
            }
        }

        MPI_Barrier(comm);
        for (i = rank*n; i < (rank+1)*n; ++i) {
            ASSERT_EQ(recvbuffer[i-rank*n], (i*(i+1))/2) << " recvbuffer[" << i-rank*n << "] is wrong on rank " << rank;
        }
    }
    MPI_Comm_free(&comm);
}

INSTANTIATE_TEST_SUITE_P(PrefixSumTestSuite,
                         PrefixSumTest,
                         testing::Values(
                            /* The first parameter is for Gradescope,
                               the second parameter is the communicator size,
                               the third parameter is the array size. */
                            std::tuple<int, int, int>{1, 2, 1},
                            std::tuple<int, int, int>{2, 4, 1},
                            std::tuple<int, int, int>{3, 4, 3},
                            std::tuple<int, int, int>{4, 8, 43},
                            std::tuple<int, int, int>{5, 8, 5000},
                            std::tuple<int, int, int>{6, 16, 2500},

                            std::tuple<int, int, int>{7, 5, 3},
                            std::tuple<int, int, int>{8, 7, 43},
                            std::tuple<int, int, int>{9, 9, 121},
                            std::tuple<int, int, int>{10, 11, 2000}
                        ));



class LeftNonzeroTest : public testing::TestWithParam< std::tuple<int, int, int> > { };

/* This operator returns the right number if it is nonzero; otherwise, it returns the first number. */
int myIntNonzero(int l, int r) { return r == 0 ? l : r; }
const void* myIntPrefixNonzero(const void* pre_v, const void* a_v, void* b_v, int len) {
    int* pre = (int*) pre_v;
    int* a = (int*) a_v;
    int* b = (int*) b_v;

    if (pre != NULL) b[0] = myIntNonzero(*pre, a[0]);
    else b[0] = a[0];

    for (int i = 1; i < len; ++i) {
        b[i] = myIntNonzero(b[i-1], a[i]);
    }
    return &b[len-1];
}

TEST_P(LeftNonzeroTest, Add)
{
    MPI_Comm comm;
    int i, n;
    int rank;
    int world_size, comm_size;
    HPC_Prefix_func *prefix_func;
    int test_number;
    char cbuf[100];

    std::tie(test_number, comm_size, n) = GetParam();

    sprintf(cbuf, "3.%d", test_number);
    RecordProperty("Points", "0.5");
    RecordProperty("Number", cbuf);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < comm_size) {
        GTEST_SKIP() << "Skipping test with comm_size=" << comm_size << " because MPI was run with " << world_size << " processes";
    }
    MPI_Comm_split(MPI_COMM_WORLD, rank < comm_size, rank, &comm);
    if (rank < comm_size) {
        MPI_Comm_rank(comm, &rank);

        prefix_func = &myIntPrefixNonzero;

        std::vector<int> sendbuffer(n), recvbuffer(n);
        for (i = rank*n; i < (rank+1)*n; ++i) sendbuffer[i-rank*n] = (i % 7 == 0 ? (i / 7 + 1) : 0);

        if (TEST_DEBUG > 0 and n < 20) {
            write_array_by_rank(comm, "sendbuffer", sendbuffer.data(), n);
            MPI_Barrier(comm);
            if (rank == 0) {
                printf("\n");
                fflush(stdout);
            }
        }


        int wb1, wb2, wb3;
        HPC_Prefix(prefix_func, &sendbuffer[0], &recvbuffer[0], n, MPI_INT, comm, &wb1, &wb2, &wb3);

        for (i = rank*n; i < (rank+1)*n; ++i) {
            ASSERT_EQ(recvbuffer[i-rank*n], i / 7 + 1) << " recvbuffer[" << i-rank*n << "] is wrong on rank " << rank;
        }
    }
    MPI_Comm_free(&comm);
}

INSTANTIATE_TEST_SUITE_P(LeftNonzeroTestSuite,
                         LeftNonzeroTest,
                         testing::Values(
                            /* The first parameter is for Gradescope,
                               the second parameter is the communicator size,
                               the third parameter is the array size. */
                            std::tuple<int, int, int>{1, 8, 1},
                            std::tuple<int, int, int>{2, 2, 10},
                            std::tuple<int, int, int>{3, 4, 15},
                            std::tuple<int, int, int>{4, 8, 8274},
                            std::tuple<int, int, int>{5, 16, 102203},

                            std::tuple<int, int, int>{6, 3, 1},
                            std::tuple<int, int, int>{7, 5, 3},
                            std::tuple<int, int, int>{8, 7, 43},
                            std::tuple<int, int, int>{9, 9, 713},
                            std::tuple<int, int, int>{10, 11, 50000}
                        ));




class Mat2x2Test : public testing::TestWithParam< std::tuple<int, int, int> > { };

typedef struct {
    int m00, m01, m10, m11;
} Mat2x2;

/* This operator adds the two matrices. */
Mat2x2 myMat2x2Add(const Mat2x2 &a, const Mat2x2 &b) {
    Mat2x2 mat;
    mat.m00 = a.m00 + b.m00;
    mat.m11 = a.m11 + b.m11;
    mat.m01 = a.m01 + b.m01;
    mat.m10 = a.m10 + b.m10;
    return mat;
}


const void* myMatAddPrefix(const void* pre_v, const void* a_v, void* b_v, int len) {
    Mat2x2* pre = (Mat2x2*) pre_v;
    Mat2x2* a = (Mat2x2*) a_v;
    Mat2x2* b = (Mat2x2*) b_v;

    if (pre != NULL) b[0] = myMat2x2Add(*pre, a[0]);
    else b[0] = a[0];

    for (int i = 1; i < len; ++i) {
        b[i] = myMat2x2Add(b[i-1], a[i]);
    }
    return &b[len-1];
}

TEST_P(Mat2x2Test, MatAdd)
{
    MPI_Comm comm;
    int i, n;
    int rank;
    int world_size, comm_size;
    HPC_Prefix_func *prefix_func;
    int test_number;
    char cbuf[100];

    std::tie(test_number, comm_size, n) = GetParam();

    sprintf(cbuf, "D.%d", test_number);
    RecordProperty("Points", "1");
    RecordProperty("Number", cbuf);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < comm_size) {
        GTEST_SKIP() << "Skipping test with comm_size=" << comm_size << " because MPI was run with " << world_size << " processes";
    }
    MPI_Comm_split(MPI_COMM_WORLD, rank < comm_size, rank, &comm);
    if (rank < comm_size) {
        MPI_Comm_rank(comm, &rank);

        MPI_Datatype mpi_mat;
        MPI_Type_contiguous(4, MPI_INT, &mpi_mat);
        MPI_Type_commit(&mpi_mat);

        prefix_func = &myMatAddPrefix;

        std::vector<Mat2x2> sendbuffer(n), recvbuffer(n);
        for (i = rank*n; i < (rank+1)*n; ++i) {
            Mat2x2 A;
            A.m00 = i;
            A.m01 = i+1;
            A.m10 = i+2;
            A.m11 = i+3;
            sendbuffer[i-rank*n] = A;
        }

        if (TEST_DEBUG > 0 and n < 20) {
            write_array_by_rank(comm, "sendbuffer", (int*) sendbuffer.data(), n*4);
            MPI_Barrier(comm);
            if (rank == 0) {
                printf("\n");
                fflush(stdout);
            }
        }


        Mat2x2 wb1, wb2, wb3;
        HPC_Prefix(prefix_func, &sendbuffer[0], &recvbuffer[0], n, mpi_mat, comm, &wb1, &wb2, &wb3);

        if (TEST_DEBUG > 0 and n < 20) {
            write_array_by_rank(comm, "recvbuffer", (int*) recvbuffer.data(), n*4);
            MPI_Barrier(comm);
            if (rank == 0) {
                printf("\n");
                fflush(stdout);
            }
        }

        for (i = rank*n; i < (rank+1)*n; ++i) {
            Mat2x2 A = recvbuffer[i-rank*n];
            ASSERT_EQ(A.m00,  (   i *(i+1))/2)      << " recvbuffer[" << i-rank*n << "].m00 is wrong on rank " << rank;
            ASSERT_EQ(A.m01,  ((i+1)*(i+2))/2)      << " recvbuffer[" << i-rank*n << "].m01 is wrong on rank " << rank;
            ASSERT_EQ(A.m10,  ((i+2)*(i+3))/2 - 1)  << " recvbuffer[" << i-rank*n << "].m10 is wrong on rank " << rank;
            ASSERT_EQ(A.m11,  ((i+3)*(i+4))/2 - 3)  << " recvbuffer[" << i-rank*n << "].m11 is wrong on rank " << rank;
        }
        MPI_Type_free(&mpi_mat);
    }
    MPI_Comm_free(&comm);
}

INSTANTIATE_TEST_SUITE_P(MatAddTestSuite,
                         Mat2x2Test,
                         testing::Values(
                            /* The first parameter is for Gradescope,
                               the second parameter is the communicator size,
                               the third parameter is the array size. */
                            std::tuple<int, int, int>{1, 4, 4}
                        ));

class BcastMatTest : public testing::TestWithParam< std::tuple<int, int, int, int> > { };

TEST_P(BcastMatTest, Bcast)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    std::vector<Mat2x2> buffer;
    int i, n;
    int world_size, comm_size;
    int root, rank;
    int test_number;
    char cbuf[100];

    std::tie(test_number, comm_size, root, n) = GetParam();

    sprintf(cbuf, "D.%d", test_number);
    RecordProperty("Points", "1");
    RecordProperty("Number", cbuf);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < comm_size) {
        GTEST_SKIP() << "Skipping test with comm_size=" << comm_size << " because MPI was run with " << world_size << " processes";
    }
    MPI_Comm_split(MPI_COMM_WORLD, rank < comm_size, rank, &comm);
    if (rank < comm_size) {
        MPI_Comm_rank(comm, &rank);

        MPI_Datatype mpi_mat;
        MPI_Type_contiguous(4, MPI_INT, &mpi_mat);
        MPI_Type_commit(&mpi_mat);

        buffer.resize(n);
        if (rank == root) {
            for (i = 0; i < n; ++i) {
                Mat2x2 A;
                A.m00 = - 2*i*i - 1;
                A.m11 =   2*i*i + 1;
                A.m01 =   2*i*i - 1;
                A.m10 = - 2*i*i + 1;
                buffer[i] = A;
            }
        }
        else {
            for (i = 0; i < n; ++i) {
                Mat2x2 A;
                A.m00 = 10000 * rank + i;
                A.m11 = 10000 * rank + 100 * i;
                A.m01 = - 10000 * rank - i;
                A.m10 = - 10000 * rank - 100 * i;
                buffer[i] = A;
            }
        }
        HPC_Bcast(&buffer[0], n, mpi_mat, root, comm);
        for (i = 0; i < n; ++i) {
            Mat2x2 A = buffer[i];
            ASSERT_EQ(A.m00, - 2*i*i - 1) << " buffer[" << i << "].m00 is wrong on rank " << rank;
            ASSERT_EQ(A.m11,   2*i*i + 1) << " buffer[" << i << "].m11 is wrong on rank " << rank;
            ASSERT_EQ(A.m01,   2*i*i - 1) << " buffer[" << i << "].m01 is wrong on rank " << rank;
            ASSERT_EQ(A.m10, - 2*i*i + 1) << " buffer[" << i << "].m10 is wrong on rank " << rank;
        }
        MPI_Type_free(&mpi_mat);
    }
    MPI_Comm_free(&comm);
}


INSTANTIATE_TEST_SUITE_P(BcastMatTestSuite,
                         BcastMatTest,
                         testing::Values(
                            /* The first parameter is for Gradescope,
                               the second parameter is the communicator size,
                               the third parameter is the root,
                               and the fourth parameter is the array size.*/
                            std::tuple<int, int, int, int>{2, 8, 0, 4}
                        ));
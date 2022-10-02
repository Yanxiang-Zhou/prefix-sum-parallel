#include "utils.h"

void write_array_by_rank(MPI_Comm comm, const char* description, const int* array, int array_len) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank != 0) {
        MPI_Recv(&rank, 1, MPI_INT, rank - 1, 0xFEEF, comm, MPI_STATUS_IGNORE);
        rank++;
    }

    printf("rank %d %s: ", rank, description);
    if (array_len > 0) printf("%2d", array[0]);
    for (int i = 1; i < array_len; ++i) {
        printf(", %2d", array[i]);
    }
    printf("\n");
    fflush(stdout);

    if (rank != size-1) {
        /* Let the next process know it's their turn to print. */
        MPI_Send(&rank, 1, MPI_INT, rank + 1, 0xFEEF, comm);
    }
}

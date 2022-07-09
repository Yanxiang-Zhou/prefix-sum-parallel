#include "collective.h"
#include "utils.h"

const int COLLECTIVE_DEBUG = 0;




/*************************** collective.h functions ************************/


 void HPC_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    // TODO: Implement this function using only sends and receives for communication instead of MPI_Bcast.
    // MPI_Bcast(buffer, count, datatype, root, comm);

    int p, rank;
    // MPI_Status status;

    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int dim = log2(p);
    
    int flip = 1 << (dim - 1);
    int mask = flip - 1;

    for (int i = dim; i > 0; i--){
        // for the processor with a smaller rank
        if (((rank ^ root) & mask) == 0){
            int flipped_proc = rank ^ flip;
            if (((rank ^ root) & flip) == 0){
                MPI_Send(buffer, count, datatype, flipped_proc, 0, comm);
            }
            else{
                MPI_Recv(buffer, count, datatype, flipped_proc, 0, comm, MPI_STATUSES_IGNORE);
            }
        }

        flip >>= 1;
        mask >>= 1;
    }
}


void HPC_Prefix(const HPC_Prefix_func* prefix_func, const void *sendbuf, void *recvbuf, int count,
                MPI_Datatype datatype, MPI_Comm comm, void* wb1, void* wb2, void* wb3) {
    if (count <= 0) return;

    /* Step 1. Run user function on local data with a NULL previous prefix. */
    const void* local_last_prefix = prefix_func(NULL, sendbuf, recvbuf, count);

    int level;
    int rank, size;
    MPI_Status status;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int vN = 1;
    while (vN < size)
        vN *= 2;

    bool init = true;
    prefix_func(NULL, local_last_prefix, wb2, 1);

    for (level = (int)log2(vN) - 1; level >= 0; level--) {
        int flipBit = (int)log2(vN) - 1 - level;
        int target = (1<<flipBit) ^ rank;

        if (target < size) {
            MPI_Send(wb2, 1, datatype, target, 111, comm);
            MPI_Recv(wb3, 1, datatype, target, 111, comm, &status);

            if (((1<<flipBit) & rank) > 0) {
                prefix_func(wb3, wb2, wb2, 1);
                if (init) {
                    prefix_func(NULL, wb3, wb1, 1);
                    init = false;
                } else {
                    prefix_func(wb3, wb1, wb1, 1);
                }
            } else {
                prefix_func(wb2, wb3, wb2, 1);
            }
        }

        MPI_Barrier(comm);
    }
    if (!init)
        prefix_func(wb1, sendbuf, recvbuf, count);
    else
        prefix_func(NULL, sendbuf, recvbuf, count);

}




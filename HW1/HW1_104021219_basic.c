#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define READ_MAX 536869888
#define IS_ODD(x) ((x) & 1)
#define IS_EVEN(x) (!(IS_ODD(x)))
#define DEBUG(x, ...) fprintf(stderr, x, __VA_ARGS__);

void swap(float *lhs, float *rhs) {
    float t = *lhs;
    *lhs = *rhs, *rhs = t;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int N = atoi(argv[1]), chunk = N/size,
        start = chunk * rank,
        bucket = chunk + (rank == size - 1) * (N - start - chunk),
        end = start + bucket;
    const char *infile = argv[2], *outfile = argv[3];
    float *res = (float *) malloc(bucket * sizeof(float)), ret;
    // File Input
    MPI_File rfile, wfile;
    MPI_Offset offset = start * sizeof(float),
        ext_offset = offset + READ_MAX * sizeof(float);
    MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &rfile);
    MPI_File_read_at_all(rfile, offset, res, bucket, MPI_FLOAT, MPI_STATUS_IGNORE);
    if (bucket >= READ_MAX) {
        MPI_File_read_at_all(rfile, ext_offset, res + READ_MAX, bucket - READ_MAX, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&rfile);
    // oesort
    enum tag { send_tag, recv_tag };
    int done, sorted;
    do {
        sorted = 1;
        // odd
        for (int i = 2 - (start & 1); i < bucket; i += 2) {
            if (res[i - 1] > res[i])
                swap(&res[i - 1], &res[i]), sorted = 0;
        }
        // even
        for (int i = 1 + (start & 1); i < bucket; i += 2) {
            if (res[i - 1] > res[i])
                swap(&res[i - 1], &res[i]), sorted = 0;
        }
    } while (!sorted);
    if (bucket != 0 && bucket != N) {
    MPI_Barrier(MPI_COMM_WORLD);
    do {
        sorted = 1;
        // odd
        for (int i = 2 - (start & 1); i < bucket; i += 2) {
            if (res[i - 1] > res[i])
                swap(&res[i - 1], &res[i]), sorted = 0;
        }
        if (rank != 0 && IS_EVEN(start)) {
            MPI_Sendrecv(&res[0], 1, MPI_FLOAT, rank - 1, send_tag,
                &ret, 1, MPI_FLOAT, rank - 1, recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ret > res[0])
                res[0] = ret, sorted = 0;
        }
        if (rank != size - 1 && IS_ODD(end)) {
            MPI_Sendrecv(&res[bucket - 1], 1, MPI_FLOAT, rank + 1, recv_tag,
                &ret, 1, MPI_FLOAT, rank + 1, send_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ret < res[bucket - 1])
                res[bucket - 1] = ret, sorted = 0;
        }
        // even
        for (int i = 1 + (start & 1); i < bucket; i += 2) {
            if (res[i - 1] > res[i])
                swap(&res[i - 1], &res[i]), sorted = 0;
        }
        if (rank != 0 && IS_ODD(start)) {
            MPI_Sendrecv(&res[0], 1, MPI_FLOAT, rank - 1, send_tag,
                &ret, 1, MPI_FLOAT, rank - 1, recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ret > res[0])
                res[0] = ret, sorted = 0;
        }
        if (rank != size - 1 && IS_EVEN(end)) {
            MPI_Sendrecv(&res[bucket - 1], 1, MPI_FLOAT, rank + 1, recv_tag,
                &ret, 1, MPI_FLOAT, rank + 1, send_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ret < res[bucket - 1])
                res[bucket - 1] = ret, sorted = 0;
        }
        MPI_Allreduce(&sorted, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    } while (!done);
    }
    // File Output
    MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &wfile);
    MPI_File_write_at_all(wfile, offset, res, bucket, MPI_FLOAT, MPI_STATUS_IGNORE);
    if (bucket >= READ_MAX) {
        MPI_File_write_at_all(wfile, ext_offset, res + READ_MAX, bucket - READ_MAX, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&wfile);
    free(res);
    MPI_Finalize();
}


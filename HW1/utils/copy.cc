#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <iostream>
#define DEBUG(x, ...) fprintf(stderr, x, __VA_ARGS__);
#define READ_MAX 536869888

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int N = atoi(argv[1]), bucket = N;
    const char *infile = argv[2], *outfile = argv[3];
    float *A = (float *) malloc(N * sizeof(float));
    // File Input
    MPI_File r1, r2, w1, w2;
    MPI_Offset o1 = 0 * sizeof(float);
    MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &r1);
    MPI_File_read_at_all(r1, o1, A, bucket, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&r1);
    MPI_Offset o2 = READ_MAX * sizeof(float);
    if (N >= READ_MAX) {
        MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &r2);
        MPI_File_read_at_all(r2, o2, A + READ_MAX, N - READ_MAX, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&r2);
    }
    // File Output
    MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &w1);
    MPI_File_write_at_all(w1, o1, A, bucket, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&w1);
    if (N >= READ_MAX) {
        MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &w2);
        MPI_File_write_at_all(w2, o2, A + READ_MAX, N - READ_MAX, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&w2);
    }
    free(A);
    MPI_Finalize();
    return 0;
}


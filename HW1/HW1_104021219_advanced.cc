#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <algorithm>
using std::sort;
#define IS_ODD(x) ((x) & 1)
#define IS_EVEN(x) (!(IS_ODD(x)))
#define DEBUG(x, ...) fprintf(stderr, x, __VA_ARGS__);

int end_merge(float *res, float *tmp, float *ret, int chunk, int bucket) {
    float *res_end = res + bucket,
        *tmp_end = tmp + bucket,
        *ret_end = ret + chunk;
    int sorted = 1;
    while (res != res_end && tmp != tmp_end && ret != ret_end) {
        if (*ret < *res) *tmp++ = *ret++, sorted = 0;
        else *tmp++ = *res++;
    }
    if (tmp != tmp_end) memcpy(tmp, res, (tmp_end - tmp) * sizeof(float));
    return sorted;
}

int front_merge(float *res, float *tmp, float *ret, int chunk, int bucket) {
    float *res_end = res + bucket - 1,
        *tmp_end = tmp + bucket - 1,
        *ret_end = ret + chunk - 1;
    int sorted = 1;
    while (res <= res_end && tmp <= tmp_end && ret <= ret_end) {
        if (*ret_end > *res_end) *tmp_end-- = *ret_end--, sorted = 0;
        else *tmp_end-- = *res_end--;
    }
    int offset = tmp_end - tmp + 1;
    if (tmp <= tmp_end) memcpy(tmp, res_end - offset + 1, offset * sizeof(float));
    return sorted;
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
    float *A = (float *) malloc(bucket * sizeof(float)),
        *B = (float *) malloc(bucket * sizeof(float)),
        *ret = (float *) malloc(chunk * sizeof(float));
    // File Input
    MPI_File rfile, wfile;
    MPI_Offset offset = start * sizeof(float);
    MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &rfile);
    MPI_File_read_at_all(rfile, offset, A, bucket, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&rfile);
    // oesort
    enum tag { send_tag, recv_tag };
    int done, sorted;
    sort(A, A + bucket);
    if (bucket != 0 && bucket != N) do {
        sorted = 1;
        // odd
        if (rank != 0 && IS_EVEN(rank)) {
            MPI_Sendrecv(A, chunk, MPI_FLOAT, rank - 1, send_tag,
                ret, chunk, MPI_FLOAT, rank - 1, recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sorted &= front_merge(A, B, ret, chunk, bucket);
        } else if (rank != size - 1 && IS_ODD(rank)) {
            MPI_Sendrecv(A, chunk, MPI_FLOAT, rank + 1, recv_tag,
                ret, chunk, MPI_FLOAT, rank + 1, send_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sorted &= end_merge(A, B, ret, chunk, bucket);
        } else std::swap(A, B);
        // even
        if (IS_ODD(rank)) {
            MPI_Sendrecv(B, chunk, MPI_FLOAT, rank - 1, send_tag,
                ret, chunk, MPI_FLOAT, rank - 1, recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sorted &= front_merge(B, A, ret, chunk, bucket);
        } else if (rank != size - 1) {
            MPI_Sendrecv(B, chunk, MPI_FLOAT, rank + 1, recv_tag,
                ret, chunk, MPI_FLOAT, rank + 1, send_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sorted &= end_merge(B, A, ret, chunk, bucket);
        } else std::swap(A, B);
        MPI_Allreduce(&sorted, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    } while (!done);
    // File Output
    MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &wfile);
    MPI_File_write_at_all(wfile, offset, A, bucket, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&wfile);
    free(A);
    free(B);
    free(ret);
    MPI_Finalize();
    return 0;
}


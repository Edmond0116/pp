#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <mpi.h>
#include <cuda_runtime.h>
#define INF 1000000000
#define V 20001
#define ceil(a, b) (1 + (((a) - 1) / (b)))
#define min(a, b) (((a) < (b)) ? (a) : (b))

// <<<1, dim3(B, B)>>>
template<int B>
__global__ void phase1(int n, int r, int *deviceDist, size_t pitch) {
    const int kmin = r * B,
        kmax = min(B, n - kmin),
        i = threadIdx.y,
        j = threadIdx.x,
        ni = kmin + i,
        nj = kmin + j;
    int t, v;
    if (ni >= n || nj >= n) return;
    __shared__ int sharedDist[B][B];
    v = sharedDist[i][j] = *((int *)((char *) deviceDist + ni * pitch) + nj);
    #pragma unroll
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        if ((t = sharedDist[i][k] + sharedDist[k][j]) < v)
            v = sharedDist[i][j] = t;
    }
    *((int *)((char *) deviceDist + ni * pitch) + nj) = v;
}
// <<<dim3(2, round), dim3(B, B)>>>
template<int B>
__global__ void phase2(int n, int r, int *deviceDist, size_t pitch) {
    if (blockIdx.y == r) return;
    const int kmin = r * B,
        kmax = min(B, n - kmin),
        i = threadIdx.y,
        j = threadIdx.x,
        ni = (blockIdx.x == 0 ? blockIdx.y : r) * B + i,
        nj = (blockIdx.x == 0 ? r : blockIdx.y) * B + j,
        ki = kmin + i,
        kj = kmin + j;
    __shared__ int sharedDist[B][B], pivotDist[B][B];
    int t, v,
        (*fromDist)[B] = blockIdx.x == 0 ? sharedDist : pivotDist,
        (*toDist)[B] = blockIdx.x == 0 ? pivotDist : sharedDist;
    v = sharedDist[i][j] = ni >= n || nj >= n ? INF :
        *((int *)((char *) deviceDist + ni * pitch) + nj);
    pivotDist[i][j] = ki >= n || kj >= n ? INF :
        *((int *)((char *) deviceDist + ki * pitch) + kj);
    if (ni >= n || nj >= n) return;
    #pragma unroll
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        if ((t = fromDist[i][k] + toDist[k][j]) < v)
            v = sharedDist[i][j] = t;
    }
    *((int *)((char *) deviceDist + ni * pitch) + nj) = v;
}
// <<<dim3(round, round), dim3(B, B)>>>
template<int B>
__global__ void phase3(int n, int r, int *deviceDist, size_t pitch, int ompIdx) {
    if (blockIdx.x == r || blockIdx.y + ompIdx == r) return;
    const int kmin = r * B,
        kmax = min(B, n - kmin),
        i = threadIdx.y,
        j = threadIdx.x,
        ni = (blockIdx.y + ompIdx) * B + i,
        nj = blockIdx.x * B + j,
        ki = kmin + i,
        kj = kmin + j;
    __shared__ int vertDist[B][B], horzDist[B][B];
    int t, v;
    vertDist[i][j] = ni >= n || kj >= n ? INF :
        *((int *)((char *) deviceDist + ni * pitch) + kj);
    horzDist[i][j] = ki >= n || nj >= n ? INF :
        *((int *)((char *) deviceDist + ki * pitch) + nj);
    if (ni >= n || nj >= n) return;
    v = *((int *)((char *) deviceDist + ni * pitch) + nj);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < kmax; ++k) {
        if ((t = vertDist[i][k] + horzDist[k][j]) < v)
            v = t;
    }
    *((int *)((char *) deviceDist + ni * pitch) + nj) = v;
}

int main(int argc, char* argv[]) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // input
    int n, m, a, b, v, *Dist, *deviceDist[2];
    std::ifstream infile(argv[1]);
    infile >> n >> m;
    cudaMallocHost((void **) &Dist, n * n * sizeof(int));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Dist[i * n + j] = (i == j ? 0 : INF);
        }
    }
    while (m--) {
        infile >> a >> b >> v;
        Dist[a * n + b] = v;
    }
    infile.close();
    // blocked FW
    size_t pitch[2];
    const int B = atoi(argv[3]), round = ceil(n, B),
            rhalf = round / 2;
    dim3 thread(B, B);
    cudaSetDevice(rank);
    cudaMallocPitch(&deviceDist[rank], &pitch[rank], n * sizeof(int), n);
    cudaMemcpy2D(deviceDist[rank], pitch[rank], Dist, n * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
    MPI_Request req;
    const int rblock = rank == 0 ? (round + 1) / 2 : rhalf;
    int kmin = 0, kmax;
    phase1<32><<<1, thread>>>(n, 0, deviceDist[rank], pitch[rank]);
    phase2<32><<<dim3(2, round), thread>>>(n, 0, deviceDist[rank], pitch[rank]);
    phase3<32><<<dim3(round, rblock), thread>>>(n, 0, deviceDist[rank], pitch[rank], rank == 0 ? rhalf : 0);
    for (int r = 1; r < round; ++r) {
        kmin += B, kmax = min(B, n - kmin);
        if (rank == 0) {
            MPI_Irecv(Dist + kmin, rhalf * B * kmax, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);
            cudaMemcpy2D(Dist + rhalf * B * n + kmin, n * sizeof(int),
                (char *) deviceDist[0] + rhalf * B * pitch[0] + kmin * sizeof(int), pitch[0],
                kmax, n - rhalf * B, cudaMemcpyDeviceToHost);
            MPI_Send(Dist + rhalf * B * n + kmin, (n - rhalf * B) * kmax, MPI_INT, 1, 0, MPI_COMM_WORLD);
            if (r < rhalf) {
                MPI_Recv(Dist + kmin * n, kmax * n, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cudaMemcpy2D((char *) deviceDist[0] + kmin * pitch[0], pitch[0],
                    Dist + kmin * n, n * sizeof(int),
                    n * sizeof(int), kmax, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy2D(Dist + kmin * n, n * sizeof(int),
                    (char *) deviceDist[0] + kmin * pitch[0], pitch[0],
                    n * sizeof(int), kmax, cudaMemcpyDeviceToHost);
                MPI_Send(Dist + kmin * n, kmax * n, MPI_INT, 1, 1, MPI_COMM_WORLD);
            }
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            cudaMemcpy2D((char *) deviceDist[0] + kmin * sizeof(int), pitch[0], Dist + kmin, n * sizeof(int),
                kmax, rhalf * B, cudaMemcpyHostToDevice);
        } else if (rhalf) {
            MPI_Irecv(Dist + rhalf * B * n + kmin, (n - rhalf * B) * kmax, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
            cudaMemcpy2D(Dist + kmin, n * sizeof(int), (char *) deviceDist[1] + kmin * sizeof(int), pitch[1],
                kmax, rhalf * B, cudaMemcpyDeviceToHost);
            MPI_Send(Dist + kmin, rhalf * B * kmax, MPI_INT, 0, 0, MPI_COMM_WORLD);
            if (r >= rhalf) {
                MPI_Recv(Dist + kmin * n, kmax * n, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cudaMemcpy2D((char *) deviceDist[1] + kmin * pitch[1], pitch[1],
                    Dist + kmin * n, n* sizeof(int),
                    n * sizeof(int), kmax, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy2D(Dist + kmin * n, n * sizeof(int),
                    (char *) deviceDist[1] + kmin * pitch[1], pitch[1],
                    n * sizeof(int), kmax, cudaMemcpyDeviceToHost);
                MPI_Send(Dist + kmin * n, kmax * n, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            cudaMemcpy2D((char *) deviceDist[1] + rhalf * B * pitch[1] + kmin * sizeof(int), pitch[1],
                Dist + rhalf * B * n + kmin, n * sizeof(int),
                kmax, n - rhalf * B, cudaMemcpyHostToDevice);
        }
        phase1<32><<<1, thread>>>(n, r, deviceDist[rank], pitch[rank]);
        phase2<32><<<dim3(2, round), thread>>>(n, r, deviceDist[rank], pitch[rank]);
        phase3<32><<<dim3(round, rblock), thread>>>(n, r, deviceDist[rank], pitch[rank], rank == 0 ? rhalf : 0);
    }
    cudaDeviceSynchronize();
    // output
    MPI_File outfile;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);
    if (rank == 0) {
        cudaMemcpy2D(
            Dist + rhalf * B * n, n * sizeof(int), (char *) deviceDist[0] + rhalf * B * pitch[0], pitch[0],
            n * sizeof(int), n - rhalf * B, cudaMemcpyDeviceToHost);
        MPI_File_write_at_all(outfile,
            rhalf * B * n * sizeof(int),
            Dist + rhalf * B * n,
            (n - rhalf * B) * n,
            MPI_INT, MPI_STATUS_IGNORE);
    } else {
        cudaMemcpy2D(Dist, n * sizeof(int), deviceDist[1], pitch[1],
            n * sizeof(int), rhalf * B, cudaMemcpyDeviceToHost);
        MPI_File_write_at_all(outfile, 0, Dist, rhalf * B * n, MPI_INT, MPI_STATUS_IGNORE);
    }
    cudaFree(deviceDist[rank]);
    MPI_Finalize();
    return 0;
}


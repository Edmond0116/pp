#include <stdio.h>
#include <stdlib.h>
#include <fstream>
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
__global__ void phase3(int n, int r, int *deviceDist, size_t pitch) {
    if (blockIdx.x == r || blockIdx.y == r) return;
    const int kmin = r * B,
        kmax = min(B, n - kmin),
        i = threadIdx.y,
        j = threadIdx.x,
        ni = blockIdx.y * B + i,
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
    int n, m, a, b, v, *Dist, *deviceDist;
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
    size_t pitch;
    cudaMallocPitch(&deviceDist, &pitch, n * sizeof(int), n);
    cudaMemcpy2D(deviceDist, pitch, Dist, n * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
    const int B = atoi(argv[3]), round = ceil(n, B);
    dim3 thread(B, B);
    #pragma unroll
    for (int r = 0; r < round; ++r) {
        phase1<32><<<1, thread>>>(n, r, deviceDist, pitch);
        phase2<32><<<dim3(2, round), thread>>>(n, r, deviceDist, pitch);
        phase3<32><<<dim3(round, round), thread>>>(n, r, deviceDist, pitch);
    }
    cudaMemcpy2D(Dist, n * sizeof(int), deviceDist, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    cudaFree(deviceDist);
    // output
    std::ofstream outfile(argv[2], std::ofstream::binary);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (Dist[i * n + j] > INF)
                Dist[i * n + j] = INF;
        }
		outfile.write(reinterpret_cast<const char *>(Dist + i * n), sizeof(int) * n);
    }
    return 0;
}


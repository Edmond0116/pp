#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <omp.h>
#include <cuda_runtime.h>
#define INF 1000000000
#define V 20001
#define ceil(a, b) (1 + (((a) - 1) / (b)))
#define min(a, b) (((a) < (b)) ? (a) : (b))

// <<<1, dim3(B, B)>>>
template<int B>
__global__ void phase1(int n, int r, int *deviceDist) {
    const int kmin = r * B,
        kmax = min(B, n - kmin),
        i = threadIdx.y,
        j = threadIdx.x,
        ni = kmin + i,
        nj = kmin + j;
    int t, v;
    if (ni >= n || nj >= n) return;
    __shared__ int sharedDist[B][B];
    v = sharedDist[i][j] = *(deviceDist + ni * n + nj);
    #pragma unroll
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        if ((t = sharedDist[i][k] + sharedDist[k][j]) < v)
            v = sharedDist[i][j] = t;
    }
    *(deviceDist + ni * n + nj) = v;
}
// <<<dim3(2, round), dim3(B, B)>>>
template<int B>
__global__ void phase2(int n, int r, int *deviceDist) {
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
    v = sharedDist[i][j] = ni >= n || nj >= n ? INF : *(deviceDist + ni * n + nj);
    pivotDist[i][j] = ki >= n || kj >= n ? INF : *(deviceDist + ki * n + kj);
    if (ni >= n || nj >= n) return;
    #pragma unroll
    for (int k = 0; k < kmax; ++k) {
        __syncthreads();
        if ((t = fromDist[i][k] + toDist[k][j]) < v)
            v = sharedDist[i][j] = t;
    }
    *(deviceDist + ni * n + nj) = v;
}
// <<<dim3(round, round), dim3(B, B)>>>
template<int B>
__global__ void phase3(int n, int r, int *deviceDist, int ompIdx) {
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
    vertDist[i][j] = ni >= n || kj >= n ? INF : *(deviceDist + ni * n + kj);
    horzDist[i][j] = ki >= n || nj >= n ? INF : *(deviceDist + ki * n + nj);
    if (ni >= n || nj >= n) return;
    v = *(deviceDist + ni * n + nj);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < kmax; ++k) {
        if ((t = vertDist[i][k] + horzDist[k][j]) < v)
            v = t;
    }
    *(deviceDist + ni * n + nj) = v;
}

int main(int argc, char* argv[]) {
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
    const int B = atoi(argv[3]), round = ceil(n, B),
            rhalf = round / 2;
    dim3 thread(B, B);
    #pragma unroll 2
    for (size_t i = 0 ; i < 2; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&deviceDist[i], n * n * sizeof(int));
        cudaMemcpy(deviceDist[i], Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    }
    #pragma omp parallel num_threads(2)
    {
        const int dev = omp_get_thread_num(),
            rblock = dev == 0 ? (round + 1) / 2 : rhalf;
        int kmin = 0, kmax;
        cudaSetDevice(dev);
        phase1<32><<<1, thread>>>(n, 0, deviceDist[dev]);
        phase2<32><<<dim3(2, round), thread>>>(n, 0, deviceDist[dev]);
        phase3<32><<<dim3(round, rblock), thread>>>(n, 0, deviceDist[dev], dev == 0 ? rhalf : 0);
        for (int r = 1; r < round; ++r) {
            kmin += B, kmax = min(B, n - kmin);
            if (dev == 0) {
                cudaMemcpyPeer(
                    deviceDist[1] + rhalf * B * n + kmin, 1,
                    deviceDist[0] + rhalf * B * n + kmin, 0,
                    kmax * (n - rhalf * B));
                if (r >= rhalf)
                    cudaMemcpyPeer(
                        deviceDist[1] + kmin * n, 1,
                        deviceDist[0] + kmin * n, 0,
                        kmax * n * sizeof(int));
            } else if (rhalf) {
               cudaMemcpyPeer(
                   deviceDist[0] + kmin, 0,
                   deviceDist[1] + kmin, 1,
                   kmax * rhalf * B);
               if (r < rhalf)
                  cudaMemcpyPeer(
                      deviceDist[0] + kmin * n, 0,
                      deviceDist[1] + kmin * n, 1,
                      kmax * n * sizeof(int));
            }
            #pragma omp barrier
            phase1<32><<<1, thread>>>(n, r, deviceDist[dev]);
            phase2<32><<<dim3(2, round), thread>>>(n, r, deviceDist[dev]);
            phase3<32><<<dim3(round, rblock), thread>>>(n, r, deviceDist[dev], dev == 0 ? rhalf : 0);
        }
        cudaDeviceSynchronize();
        if (dev == 0)
            cudaMemcpy(
                Dist + rhalf * B * n, deviceDist[0] + rhalf * B * n,
                (n - rhalf * B) * n * sizeof(int), cudaMemcpyDeviceToHost);
        else
            cudaMemcpy(Dist, deviceDist[1], rhalf * B * n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(deviceDist[dev]);
        #pragma omp barrier
    }
    // output
    std::ofstream outfile(argv[2], std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(Dist), n * n * sizeof(int));
    return 0;
}


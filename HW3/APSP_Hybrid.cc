#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <pthread.h>
#define INF 0x3f3f3f3f
#define DEBUG 1
#define DBG(x, ...) if (DEBUG) { fprintf(stderr, x, __VA_ARGS__); }

int n, nc, *nb, *nbi, *cost, relaxed[2];
pthread_mutex_t mtx;
void *dijk(void *arg) {
    int ret = 1, w, *d = (int *) arg;
    bool *used = (bool *) malloc(sizeof(bool) * n);
    for (int i = 0; i < n; ++i) used[i] = false;
    while (true) {
        int v = -1;
        for (int u = 0; u < n; ++u)
            if (!used[u] && (v == -1 || d[u] < d[v])) v = u;
        if (v == -1) break;
        used[v] = true;
        for (int u = 0; u < n; ++u)
            if ((w = d[v] + cost[v * n + u]) < d[u])
                d[u] = w, ret = 0;
    }
    free(used);
    pthread_mutex_lock(&mtx);
    relaxed[0] &= ret;
    pthread_mutex_unlock(&mtx);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // input
    int m;//, N = atoi(argv[3]);
    FILE *infile = fopen(argv[1], "r");
    fscanf(infile, "%d%d", &n, &m);
    int *nstarts = (int *) malloc(sizeof(int) * size),
        *nbuckets = (int *) malloc(sizeof(int) * size);
    for (int i = 0; i < size; ++i)
        nbuckets[i] = n * (n / size + (i == size - 1 ? n % size : 0)),
        nstarts[i] = i == 0 ? 0 : nstarts[i - 1] + nbuckets[i - 1];
    const int N = n / size,
        start = rank * N,
        end = rank == size - 1 ? n : (rank + 1) * N,
        bucket = end - start;
    int *d = (int *) malloc(sizeof(int) * bucket * n);
    cost = (int *) malloc(sizeof(int) * n * n);
    for (int i = 0; i < n * n; ++i)
        cost[i] = (i/n == i%n ? 0 : INF);
    for (int i = 0; i < bucket; ++i)
        for (int j = 0; j < n; ++j)
            d[i * n + j] = (j == start + i ? 0 : INF);
    int a, b, w;
    for (int i = 0; i < m; ++i) {
        fscanf(infile, "%d%d%d", &a, &b, &w);
        if (a >= start && a < end)
            d[(a - start) * n + b] = cost[a * n + b] = w;
        if (b >= start && b < end)
            d[(b - start) * n + a] = cost[b * n + a] = w;
    }
    fclose(infile);
    // neighbor
    nc = 0;
    nb = (int *) malloc(sizeof(int) * size);
    nbi = (int *) malloc(sizeof(int) * size);
    memset(nbi, -1, sizeof(int) * size);
    for (int i = 0; i < bucket; ++i)
        for (int j = 0; j < n; ++j)
            if (j / N != rank && d[i * n + j] != INF && nbi[j / N] == -1)
                nb[nc] = j / N, nbi[j / N] = nc, nc++;
    // apsp
    enum tag { d_tag, t_tag };
    int done[2];
    pthread_mutex_init(&mtx, NULL);
    pthread_t *threads;
    do {
        for (int i = 0; i < nc; ++i)
            MPI_Sendrecv(d, bucket * n, MPI_INT, nb[i], d_tag,
                cost + nstarts[nb[i]], nbuckets[nb[i]],
                MPI_INT, nb[i], d_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        relaxed[0] = 1;
        // pthread
        threads = (pthread_t *) malloc(sizeof(pthread_t) * bucket);
        for (int i = 0; i < bucket; ++i)
            pthread_create(&threads[i], NULL, dijk, d + i * n);
        for (int i = 0; i < bucket; ++i)
            pthread_join(threads[i], NULL);
        free(threads);
        MPI_Sendrecv(&relaxed, 2, MPI_INT, (rank + 1)%size, t_tag,
            &done, 2, MPI_INT, (rank + size - 1)%size, t_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (!relaxed[0] || !done[0]) relaxed[1] = 0;
        else relaxed[1] = 1 + (relaxed[1] > done[1] ? done[1] : relaxed[1]);
    } while (relaxed[1] < 2 * size);
    pthread_mutex_destroy(&mtx);
    MPI_Gatherv(d, bucket * n, MPI_INT, cost, nbuckets, nstarts, MPI_INT, 0, MPI_COMM_WORLD);
    // output
    if (rank == 0) {
        FILE *outfile = fopen(argv[2], "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(outfile, "%d%s",
                    (i == j ? 0 : cost[i * n + j]),
                    (j == n - 1 ? " \n" : " ")
                );
            }
        }
    }
    free(nstarts);
    free(nbuckets);
    free(d);
    free(cost);
    free(nb);
    free(nbi);
    MPI_Finalize();
}

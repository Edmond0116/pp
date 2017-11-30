#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define INF 0x3f3f3f3f
#define DEBUG 1
#define DBG(x, ...) if (DEBUG) { fprintf(stderr, x, __VA_ARGS__); }

bool *used;
int n, *d, nc, *nb, *nbi, *cost;
int dijk() {
    int ret = 1, w;
    for (int i = 0; i < n; ++i) used[i] = (nbi[i] == -1);
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
    return ret;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // input
    int m; //, N = atoi(argv[3]);
    FILE *infile = fopen(argv[1], "r");
    fscanf(infile, "%d%d", &n, &m);
    d = (int *) malloc(sizeof(int) * n);
    cost = (int *) malloc(sizeof(int) * n * n);
    used = (bool *) malloc(sizeof(bool) * n);
    for (int i = 0; i < n * n; ++i) {
        cost[i] = (i/n == i%n ? 0 : INF);
    }
    for (int i = 0; i < n; ++i) d[i] = (i == rank ? 0 : INF);
    int a, b, w;
    for (int i = 0; i < m; ++i) {
        fscanf(infile, "%d%d%d", &a, &b, &w);
        if (a == rank) d[b] = cost[a * n + b] = w;
        if (b == rank) d[a] = cost[b * n + a] = w;
    }
    fclose(infile);
    // neighbor
    nc = 0;
    nb = (int *) malloc(sizeof(int) * n);
    nbi = (int *) malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i)
        if (i != rank && d[i] != INF) nb[nc] = i, nbi[i] = nc, nc++;
        else nbi[i] = -1;
    nbi[rank] = nc; nb[nc] = rank;
    // apsp
    enum tag { d_tag };
    int done, relaxed;
    do {
        for (int i = 0; i < nc; ++i)
            MPI_Sendrecv(d, n, MPI_INT, nb[i], d_tag,
                cost + n * nb[i], n, MPI_INT, nb[i], d_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        relaxed = dijk();
        MPI_Allreduce(&relaxed, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    } while (!done);
    MPI_Gather(d, n, MPI_INT, cost, n, MPI_INT, 0, MPI_COMM_WORLD);
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
    free(d);
    free(cost);
    free(used);
    free(nb);
    free(nbi);
    MPI_Finalize();
}

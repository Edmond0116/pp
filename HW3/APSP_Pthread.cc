#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#define INF 0x3f3f3f3f
#define DEBUG 1
#define DBG(x, ...) if (DEBUG) { fprintf(stderr, x, __VA_ARGS__); }

pthread_barrier_t barrier;
int n, m, N, *d;
// FW
void *FW(void *arg) {
    int w, id = *((int *) arg),
        start = id * n / N,
        end = (id + 1) * n / N;
    for (int k = 0; k < n; ++k) {
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < n; ++j)
                if ((w = d[i * n + k] + d[k * n + j]) < d[i * n + j])
                    d[i * n + j] = w;
        }
        pthread_barrier_wait(&barrier);
    }
}

int main(int argc, char** argv) {
    N = atoi(argv[3]);
    // input
    FILE *infile = fopen(argv[1], "r");
    fscanf(infile, "%d%d", &n, &m);
    d = (int *) malloc(sizeof(int *) * n * n);
    for (int i = 0; i < n * n; ++i) d[i] = INF;
    int a, b, w;
    for (int i = 0; i < m; ++i) {
        fscanf(infile, "%d%d%d", &a, &b, &w);
        d[a * n + b] = d[b * n + a] = w;
    }
    fclose(infile);
    // pthread
    pthread_barrier_init(&barrier, NULL, N);
    pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * N);
    for (int i = 0; i < N; ++i)
        pthread_create(&threads[i], NULL, FW, new int(i));
    for (int i = 0; i < N; ++i)
        pthread_join(threads[i], NULL);
    pthread_barrier_destroy(&barrier);
    // ouput
    FILE *outfile = fopen(argv[2], "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(outfile, "%d%s",
                (i == j ? 0 : d[i * n + j]),
                (j == n - 1 ? " \n" : " ")
            );
        }
    }
    free(d);
}

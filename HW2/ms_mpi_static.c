#define PNG_NO_SETJMP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <mpi.h>
#define MAX_ITER 100000
#define MAX_NP 128

int check(double x0, double y0) {
    double x = x0 - 0.25,
        x2 = x * x,
        y2 = y0 * y0,
        t = x2 + y2 + x / 2;
    if (t * t - (x2 + y2) / 4 < 0) return MAX_ITER;
    if ((x0 + 1) * (x0 + 1) + y2 < 0.0625) return MAX_ITER;
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int width = strtol(argv[6], 0, 10),
        height = strtol(argv[7], 0, 10),
        wh = width * height,
        chunk = wh / size;
    const double left = strtod(argv[2], 0),
        right = strtod(argv[3], 0),
        lower = strtod(argv[4], 0),
        upper = strtod(argv[5], 0);
    const char* filename = argv[8];
    /* mpi */
    int buckets[MAX_NP] = {chunk + wh % size}, starts[MAX_NP] = {};
    for (int i = 1; i < size; ++i)
        buckets[i] = chunk,
        starts[i] = starts[i - 1] + buckets[i - 1];
    const int start = starts[rank],
        bucket = buckets[rank],
        end = start + bucket;
    int *image = (int *) malloc(bucket * sizeof(int)), *result;
    if (rank == 0) result = (int *) malloc(wh * sizeof(int));
    /* mandelbrot set */
    double x, y, x0, y0, xn, yn,
        xstep = (right - left) / width,
        ystep = (upper - lower) / height;
    for (int k = start; k < end; ++k) {
        x0 = left + (k % width) * xstep;
        y0 = lower + (k / width) * ystep;
        int iter = check(x0, y0);
        for (x = y = 0; iter < MAX_ITER;) {
            xn = x * x - y * y + x0;
            yn = 2 * x * y + y0;
            ++iter;
            if (x == xn && y == yn) {
                iter = MAX_ITER;
                break;
            } else if (xn * xn + yn * yn >= 4) break;
            x = xn, y = yn;
        }
        image[k - start] = iter;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(image, bucket, MPI_INT, result, buckets, starts, MPI_INT, 0, MPI_COMM_WORLD);
    /* write png */
    if (rank == 0) {
        FILE* fp = fopen(filename, "wb");
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_infop info_ptr = png_create_info_struct(png_ptr);
        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);
        size_t row_size = 3 * width * sizeof(png_byte);
        png_bytep row = (png_bytep) malloc(row_size);
        for (int y = height - 1; y >= 0; --y) {
            memset(row, 0, row_size);
            for (int x = 0; x < width; ++x) {
                row[x * 3] = ((result[y * width + x] & 0xf) << 4);
            }
            png_write_row(png_ptr, row);
        }
        free(row);
        png_write_end(png_ptr, NULL);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        free(result);
    }
    free(image);
    MPI_Finalize();
}


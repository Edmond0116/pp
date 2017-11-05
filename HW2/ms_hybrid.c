#define PNG_NO_SETJMP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <mpi.h>
#include <omp.h>
#define MAX_ITER 100000
#define MAX_NP 128
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define DEBUG(x, ...) fprintf(stderr, x, __VA_ARGS__);

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
    const int num_threads = strtol(argv[1], 0, 10),
        width = strtol(argv[6], 0, 10),
        height = strtol(argv[7], 0, 10),
        wh = width * height,
        chunk = size == 1 ? wh : 256 * num_threads,
        has_master = size != 1;
    const double left = strtod(argv[2], 0),
        right = strtod(argv[3], 0),
        lower = strtod(argv[4], 0),
        upper = strtod(argv[5], 0),
        xstep = (right - left) / width,
        ystep = (upper - lower) / height;
    const char* filename = argv[8];
    enum tag { send_tag, recv_tag };
    int start = 0, *image = (int *) malloc(chunk * sizeof(int)), *result;
    /* mandelbrot set */
    if (rank == 0 && has_master) {
        result = (int *) malloc((wh + chunk) * sizeof(int));
        MPI_Request calc[MAX_NP];
        int wait[MAX_NP], wait_size = size - 1, r;
        for (int i = 0; i < size - 1; ++i) wait[i] = i, calc[i] = MPI_REQUEST_NULL;
        do {
            for (int i = 0; i < wait_size && start < wh; ++i) {
                r = wait[i] + 1;
                MPI_Send(&start, 1, MPI_INT, r, send_tag, MPI_COMM_WORLD);
                MPI_Irecv(result + start, chunk, MPI_INT, r, recv_tag, MPI_COMM_WORLD, calc + r - 1);
                start += chunk;
            }
            MPI_Testsome(size - 1, calc, &wait_size, wait, MPI_STATUSES_IGNORE);
        } while (start < wh);
        for (int i = 0; i < size - 1; ++i)
            MPI_Send(&wh, 1, MPI_INT, i + 1, send_tag, MPI_COMM_WORLD);
        MPI_Waitall(size - 1, calc, MPI_STATUSES_IGNORE);
    } else {
        do {
            if (has_master) MPI_Recv(&start, 1, MPI_INT, 0, send_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (start >= wh) break;
            const int end = MIN(start + chunk, wh);
            #pragma omp num_thread(num_threads)
            #pragma omp parallel for schedule(dynamic)
            for (int k = start; k < end; ++k) {
                double x, y, x0, y0, xn, yn;
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
            if (has_master) MPI_Send(image, chunk, MPI_INT, 0, recv_tag, MPI_COMM_WORLD);
        } while (has_master);
    }
    if (!has_master) result = image;
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
            #pragma omp parallel for
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
    if (has_master) free(image);
    MPI_Finalize();
}


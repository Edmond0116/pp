#define PNG_NO_SETJMP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <omp.h>
#define MAX_ITER 100000

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
    const int num_threads = strtol(argv[1], 0, 10),
        width = strtol(argv[6], 0, 10),
        height = strtol(argv[7], 0, 10);
    const double left = strtod(argv[2], 0),
        right = strtod(argv[3], 0),
        lower = strtod(argv[4], 0),
        upper = strtod(argv[5], 0);
    const char* filename = argv[8];
    int *image = (int *) malloc(width * height * sizeof(int));
    /* mandelbrot set */
    double x, y, x0, y0, xn, yn,
        xstep = (right - left) / width,
        ystep = (upper - lower) / height;
    #pragma omp num_thread(num_threads)
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            x0 = left + i * xstep;
            y0 = lower + j * ystep;
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
            image[j * width + i] = iter;
        }
    }
    /* write png */
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
            row[x * 3] = ((image[y * width + x] & 0xf) << 4);
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    free(image);
}


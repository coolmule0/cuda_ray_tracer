#include <iostream>
#include <time.h>

#include "color.h"
#include "vec3.h"

int main() {

    // Image

    int image_width = 256;
    int image_height = 256;

    // Render

    clock_t start, stop;
    start = clock();

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto pixel_color = color(double(i)/(image_width-1), double(j)/(image_height-1), 0);
            write_color(std::cout, pixel_color);
        }
    }
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

    std::clog << "\rDone.                 \n";
    std::cerr << "took " << timer_seconds << " seconds.\n";
}
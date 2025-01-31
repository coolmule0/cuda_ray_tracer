#ifndef COLOR_H
#define COLOR_H

#include "interval.cuh"
#include "vec3.cuh"

using color = vec3;

inline double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}

// Takes a [0,1] range vec3/color and prints out a 255 range
void write_color(std::ostream& out, color* fb, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;

            auto r = fb[pixel_index].r();
            auto g = fb[pixel_index].g();
            auto b = fb[pixel_index].b();

            // Apply a linear to gamma transform for gamma 2
            r = linear_to_gamma(r);
            g = linear_to_gamma(g);
            b = linear_to_gamma(b);

            // Translate the [0,1] component values to the byte range [0,255].
            static const interval intensity(0.000, 0.999);
            int rbyte = int(256 * intensity.clamp(r));
            int gbyte = int(256 * intensity.clamp(g));
            int bbyte = int(256 * intensity.clamp(b));
            
            // Write out the pixel color components.
            out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
        }
    }
}

#endif